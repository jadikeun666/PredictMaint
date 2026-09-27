"""
Stock-effect tests for the transition endpoint — inventory.md §2.
Separate module from tests.py (Giliran 1's pure state-machine matrix) so
each stays focused: that file asserts transition validity/role-gate/
required-fields, this one asserts the inventory side effects layered on
top in Giliran 2.
"""
from datetime import timedelta

from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient
from rest_framework_simplejwt.tokens import RefreshToken

from apps.accounts.models import User
from apps.alerting.models import Alert
from apps.inventory.models import SparePart, StockMovement
from apps.registry.models import Bearing, Machine
from apps.scheduling.models import WorkOrder, WorkOrderPart


def auth_header(user):
    token = RefreshToken.for_user(user)
    return {"HTTP_AUTHORIZATION": f"Bearer {token.access_token}"}


class StockEffectTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.machine = Machine.objects.create(
            name="Test Machine", machine_type="pump", criticality=Machine.Criticality.MEDIUM
        )
        cls.bearing = Bearing.objects.create(
            machine=cls.machine, position_label="drive_end", bearing_model="SKF 6205-2RS"
        )
        cls.admin = User.objects.create_user(username="admin2", password="x", role=User.Role.ADMIN)
        cls.technician = User.objects.create_user(username="tech2", password="x", role=User.Role.TECHNICIAN)

    def make_alert(self):
        now = timezone.now()
        return Alert.objects.create(
            bearing=self.bearing,
            alert_type=Alert.AlertType.FAULT_DETECTED,
            severity=Alert.Severity.CRITICAL,
            triggered_at=now,
            last_seen_at=now,
        )

    def make_wo_draft_with_part(self, on_hand, required):
        spare = SparePart.objects.create(
            sku=f"SKU-{on_hand}-{required}-{SparePart.objects.count()}",
            name="Test Part",
            quantity_on_hand=on_hand,
            reorder_point=1,
            reorder_quantity=5,
        )
        alert = self.make_alert()
        wo = WorkOrder.objects.create(
            bearing=self.bearing, alert=alert, status=WorkOrder.Status.DRAFT, priority=WorkOrder.Priority.URGENT
        )
        part = WorkOrderPart.objects.create(
            work_order=wo, spare_part=spare, quantity_required=required, quantity_reserved=0
        )
        return wo, spare, part

    def schedule_body(self):
        now = timezone.now()
        return {
            "to_status": WorkOrder.Status.SCHEDULED,
            "scheduled_start": now.isoformat(),
            "scheduled_end": (now + timedelta(hours=4)).isoformat(),
        }

    def test_sufficient_stock_reserves_without_decrementing(self):
        wo, spare, part = self.make_wo_draft_with_part(on_hand=10, required=2)
        client = APIClient()
        client.credentials(**auth_header(self.admin))
        resp = client.post(f"/api/v1/work-orders/{wo.id}/transition/", self.schedule_body(), format="json")
        self.assertEqual(resp.status_code, 200)

        part.refresh_from_db()
        spare.refresh_from_db()
        self.assertEqual(part.quantity_reserved, 2)
        self.assertEqual(spare.quantity_on_hand, 10, "on-hand must NOT decrement at reservation time")
        self.assertNotIn("Insufficient stock", resp.data["notes"])

    def test_insufficient_stock_warns_but_still_schedules(self):
        wo, spare, part = self.make_wo_draft_with_part(on_hand=1, required=5)
        client = APIClient()
        client.credentials(**auth_header(self.admin))
        resp = client.post(f"/api/v1/work-orders/{wo.id}/transition/", self.schedule_body(), format="json")
        self.assertEqual(resp.status_code, 200, "insufficient stock must not block the transition")

        part.refresh_from_db()
        self.assertEqual(part.quantity_reserved, 0)
        self.assertIn(f"Insufficient stock for {spare.sku}: have 1, need 5", resp.data["notes"])

    def test_completed_issues_stock_and_resolves_alert(self):
        wo, spare, part = self.make_wo_draft_with_part(on_hand=10, required=3)
        client = APIClient()
        client.credentials(**auth_header(self.admin))

        resp = client.post(f"/api/v1/work-orders/{wo.id}/transition/", self.schedule_body(), format="json")
        self.assertEqual(resp.status_code, 200)

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

        spare.refresh_from_db()
        self.assertEqual(spare.quantity_on_hand, 7, "10 on-hand - 3 reserved/issued = 7")

        movements = StockMovement.objects.filter(spare_part=spare, reference_work_order=wo)
        self.assertEqual(movements.count(), 1)
        movement = movements.first()
        self.assertEqual(movement.movement_type, StockMovement.MovementType.ISSUE)
        self.assertEqual(movement.quantity, -3)

        wo.refresh_from_db()
        wo.alert.refresh_from_db()
        self.assertIsNotNone(wo.alert.resolved_at)
        self.assertIsNone(wo.alert.resolved_by)

    def test_cancelled_releases_reservation_without_stock_movement(self):
        wo, spare, part = self.make_wo_draft_with_part(on_hand=10, required=4)
        client = APIClient()
        client.credentials(**auth_header(self.admin))

        resp = client.post(f"/api/v1/work-orders/{wo.id}/transition/", self.schedule_body(), format="json")
        self.assertEqual(resp.status_code, 200)
        part.refresh_from_db()
        self.assertEqual(part.quantity_reserved, 4)

        resp = client.post(
            f"/api/v1/work-orders/{wo.id}/transition/",
            {"to_status": WorkOrder.Status.CANCELLED, "notes": "customer cancelled job"},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)

        part.refresh_from_db()
        spare.refresh_from_db()
        self.assertEqual(part.quantity_reserved, 0)
        self.assertEqual(spare.quantity_on_hand, 10, "on-hand untouched — reservation was never physical")
        self.assertEqual(StockMovement.objects.filter(spare_part=spare).count(), 0)
