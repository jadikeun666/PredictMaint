"""
Work Order status state machine tests — scheduling.md §4,
engineering-rules.md §Testing: "every listed transition succeeds, every
non-listed transition is rejected with HTTP 409, as a single
parametrized test matrix, not ad-hoc individual tests."

Giliran 1 scope only: transition validity + role gate + required-field
validation via the HTTP endpoint. Stock/alert side effects (Giliran 2)
are NOT asserted here — a separate test module covers those once that
code lands, per this session's plan.
"""
from datetime import timedelta

from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient
from rest_framework_simplejwt.tokens import RefreshToken

from apps.accounts.models import User
from apps.registry.models import Bearing, Machine
from apps.scheduling.models import WorkOrder

ALL_STATUSES = [c[0] for c in WorkOrder.Status.choices]

# Mirrors apps.scheduling.transitions.TRANSITIONS keys exactly — kept as a
# separate literal here (not imported) so the test independently encodes
# scheduling.md §4 rather than just re-checking whatever the code happens
# to contain.
VALID_TRANSITIONS = {
    (WorkOrder.Status.DRAFT, WorkOrder.Status.SCHEDULED),
    (WorkOrder.Status.DRAFT, WorkOrder.Status.CANCELLED),
    (WorkOrder.Status.SCHEDULED, WorkOrder.Status.CANCELLED),
    (WorkOrder.Status.SCHEDULED, WorkOrder.Status.IN_PROGRESS),
    (WorkOrder.Status.IN_PROGRESS, WorkOrder.Status.COMPLETED),
    (WorkOrder.Status.IN_PROGRESS, WorkOrder.Status.CANCELLED),
}


def auth_header(user):
    token = RefreshToken.for_user(user)
    return {"HTTP_AUTHORIZATION": f"Bearer {token.access_token}"}


class WorkOrderTransitionMatrixTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.machine = Machine.objects.create(
            name="Test Machine",
            machine_type="pump",
            criticality=Machine.Criticality.MEDIUM,
        )
        cls.bearing = Bearing.objects.create(
            machine=cls.machine,
            position_label="drive_end",
            bearing_model="SKF 6205-2RS",
        )
        cls.admin = User.objects.create_user(username="admin1", password="x", role=User.Role.ADMIN)
        cls.engineer = User.objects.create_user(username="eng1", password="x", role=User.Role.ENGINEER)
        cls.technician = User.objects.create_user(username="tech1", password="x", role=User.Role.TECHNICIAN)
        cls.viewer = User.objects.create_user(username="view1", password="x", role=User.Role.VIEWER)

    def make_wo(self, status):
        now = timezone.now()
        wo = WorkOrder.objects.create(
            bearing=self.bearing,
            status=status,
            priority=WorkOrder.Priority.URGENT,
        )
        if status in (WorkOrder.Status.SCHEDULED, WorkOrder.Status.IN_PROGRESS, WorkOrder.Status.COMPLETED):
            wo.scheduled_start = now
            wo.scheduled_end = now + timedelta(hours=4)
            wo.save()
        return wo

    def body_for(self, to_status):
        if to_status == WorkOrder.Status.SCHEDULED:
            now = timezone.now()
            return {
                "to_status": to_status,
                "scheduled_start": now.isoformat(),
                "scheduled_end": (now + timedelta(hours=4)).isoformat(),
            }
        if to_status == WorkOrder.Status.CANCELLED:
            return {"to_status": to_status, "notes": "test cancel reason"}
        return {"to_status": to_status}

    def test_every_from_to_pair_as_admin(self):
        """
        Full 5x5 matrix as ADMIN (a role permitted on every valid pair) so
        this test isolates "is (from, to) in the state machine at all" from
        role-gating, which gets its own tests below.
        """
        client = APIClient()
        client.credentials(**auth_header(self.admin))
        for from_status in ALL_STATUSES:
            for to_status in ALL_STATUSES:
                with self.subTest(from_status=from_status, to_status=to_status):
                    wo = self.make_wo(from_status)
                    resp = client.post(
                        f"/api/v1/work-orders/{wo.id}/transition/",
                        self.body_for(to_status),
                        format="json",
                    )
                    if (from_status, to_status) in VALID_TRANSITIONS:
                        self.assertEqual(
                            resp.status_code,
                            200,
                            f"{from_status}->{to_status} should succeed for ADMIN, "
                            f"got {resp.status_code}: {resp.data}",
                        )
                    else:
                        self.assertEqual(
                            resp.status_code,
                            409,
                            f"{from_status}->{to_status} should be 409, "
                            f"got {resp.status_code}: {resp.data}",
                        )

    def test_technician_cannot_cancel_in_progress(self):
        # scheduling.md §4: IN_PROGRESS -> CANCELLED is ENGINEER/ADMIN only.
        wo = self.make_wo(WorkOrder.Status.IN_PROGRESS)
        client = APIClient()
        client.credentials(**auth_header(self.technician))
        resp = client.post(
            f"/api/v1/work-orders/{wo.id}/transition/",
            {"to_status": WorkOrder.Status.CANCELLED, "notes": "x"},
            format="json",
        )
        self.assertEqual(resp.status_code, 403)

    def test_technician_can_start_and_complete(self):
        wo = self.make_wo(WorkOrder.Status.SCHEDULED)
        client = APIClient()
        client.credentials(**auth_header(self.technician))
        resp = client.post(
            f"/api/v1/work-orders/{wo.id}/transition/",
            {"to_status": WorkOrder.Status.IN_PROGRESS},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        resp = client.post(
            f"/api/v1/work-orders/{wo.id}/transition/",
            {"to_status": WorkOrder.Status.COMPLETED},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)

    def test_viewer_cannot_transition_anything(self):
        wo = self.make_wo(WorkOrder.Status.DRAFT)
        client = APIClient()
        client.credentials(**auth_header(self.viewer))
        now = timezone.now()
        resp = client.post(
            f"/api/v1/work-orders/{wo.id}/transition/",
            {
                "to_status": WorkOrder.Status.SCHEDULED,
                "scheduled_start": now.isoformat(),
                "scheduled_end": (now + timedelta(hours=4)).isoformat(),
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 403)

    def test_missing_required_field_for_scheduled_is_400(self):
        wo = self.make_wo(WorkOrder.Status.DRAFT)
        client = APIClient()
        client.credentials(**auth_header(self.admin))
        resp = client.post(
            f"/api/v1/work-orders/{wo.id}/transition/",
            {"to_status": WorkOrder.Status.SCHEDULED},
            format="json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_missing_notes_for_in_progress_cancel_is_400(self):
        wo = self.make_wo(WorkOrder.Status.IN_PROGRESS)
        client = APIClient()
        client.credentials(**auth_header(self.admin))
        resp = client.post(
            f"/api/v1/work-orders/{wo.id}/transition/",
            {"to_status": WorkOrder.Status.CANCELLED},
            format="json",
        )
        self.assertEqual(resp.status_code, 400)
