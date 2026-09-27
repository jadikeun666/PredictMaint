"""
inventory.md §3 (RECEIPT), §4 (LOW_STOCK), §6 (ADJUSTMENT) tests, plus
api.md role gates for /spare-parts/ and /stock-movements/.
"""
from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient
from rest_framework_simplejwt.tokens import RefreshToken

from apps.accounts.models import User
from apps.alerting.models import Alert
from apps.inventory.models import SparePart, StockMovement


def auth_header(user):
    token = RefreshToken.for_user(user)
    return {"HTTP_AUTHORIZATION": f"Bearer {token.access_token}"}


class InventoryEndpointTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.admin = User.objects.create_user(username="admin3", password="x", role=User.Role.ADMIN)
        cls.engineer = User.objects.create_user(username="eng3", password="x", role=User.Role.ENGINEER)
        cls.technician = User.objects.create_user(username="tech3", password="x", role=User.Role.TECHNICIAN)
        cls.viewer = User.objects.create_user(username="view3", password="x", role=User.Role.VIEWER)

    def make_part(self, on_hand, reorder_point=5):
        return SparePart.objects.create(
            sku=f"SKU-{SparePart.objects.count()}",
            name="Test Part",
            quantity_on_hand=on_hand,
            reorder_point=reorder_point,
            reorder_quantity=10,
        )

    # --- SparePart role gate ---

    def test_admin_can_create_spare_part(self):
        client = APIClient()
        client.credentials(**auth_header(self.admin))
        resp = client.post(
            "/api/v1/spare-parts/",
            {"sku": "NEW-1", "name": "New Part", "quantity_on_hand": 0, "reorder_point": 1, "reorder_quantity": 5},
            format="json",
        )
        self.assertEqual(resp.status_code, 201, resp.data)

    def test_engineer_cannot_create_spare_part(self):
        client = APIClient()
        client.credentials(**auth_header(self.engineer))
        resp = client.post(
            "/api/v1/spare-parts/",
            {"sku": "NEW-2", "name": "New Part", "quantity_on_hand": 0, "reorder_point": 1, "reorder_quantity": 5},
            format="json",
        )
        self.assertEqual(resp.status_code, 403)

    def test_quantity_on_hand_is_read_only_on_patch(self):
        part = self.make_part(on_hand=5)
        client = APIClient()
        client.credentials(**auth_header(self.admin))
        resp = client.patch(f"/api/v1/spare-parts/{part.id}/", {"quantity_on_hand": 999}, format="json")
        self.assertEqual(resp.status_code, 200)
        part.refresh_from_db()
        self.assertEqual(part.quantity_on_hand, 5, "quantity_on_hand must only change via StockMovement")

    # --- StockMovement role gate + validation ---

    def test_technician_cannot_create_stock_movement(self):
        part = self.make_part(on_hand=5)
        client = APIClient()
        client.credentials(**auth_header(self.technician))
        resp = client.post(
            "/api/v1/stock-movements/",
            {"spare_part": str(part.id), "movement_type": "RECEIPT", "quantity": 5},
            format="json",
        )
        self.assertEqual(resp.status_code, 403)

    def test_issue_rejected_via_endpoint(self):
        part = self.make_part(on_hand=5)
        client = APIClient()
        client.credentials(**auth_header(self.engineer))
        resp = client.post(
            "/api/v1/stock-movements/",
            {"spare_part": str(part.id), "movement_type": "ISSUE", "quantity": -1},
            format="json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_adjustment_without_reason_is_400(self):
        part = self.make_part(on_hand=5)
        client = APIClient()
        client.credentials(**auth_header(self.engineer))
        resp = client.post(
            "/api/v1/stock-movements/",
            {"spare_part": str(part.id), "movement_type": "ADJUSTMENT", "quantity": -2},
            format="json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_receipt_positive_quantity_required(self):
        part = self.make_part(on_hand=5)
        client = APIClient()
        client.credentials(**auth_header(self.engineer))
        resp = client.post(
            "/api/v1/stock-movements/",
            {"spare_part": str(part.id), "movement_type": "RECEIPT", "quantity": -3},
            format="json",
        )
        self.assertEqual(resp.status_code, 400)

    # --- Side effects: quantity_on_hand + LOW_STOCK ---

    def test_receipt_increments_quantity_on_hand(self):
        part = self.make_part(on_hand=5, reorder_point=2)
        client = APIClient()
        client.credentials(**auth_header(self.engineer))
        resp = client.post(
            "/api/v1/stock-movements/",
            {"spare_part": str(part.id), "movement_type": "RECEIPT", "quantity": 10},
            format="json",
        )
        self.assertEqual(resp.status_code, 201, resp.data)
        part.refresh_from_db()
        self.assertEqual(part.quantity_on_hand, 15)

    def test_adjustment_below_reorder_point_creates_low_stock_alert(self):
        part = self.make_part(on_hand=10, reorder_point=5)
        client = APIClient()
        client.credentials(**auth_header(self.engineer))
        resp = client.post(
            "/api/v1/stock-movements/",
            {"spare_part": str(part.id), "movement_type": "ADJUSTMENT", "quantity": -7, "reason": "cycle count"},
            format="json",
        )
        self.assertEqual(resp.status_code, 201, resp.data)
        part.refresh_from_db()
        self.assertEqual(part.quantity_on_hand, 3)

        alerts = Alert.objects.filter(alert_type=Alert.AlertType.LOW_STOCK, spare_part_id=part.id)
        self.assertEqual(alerts.count(), 1)
        self.assertEqual(alerts.first().severity, Alert.Severity.WARNING)
        self.assertIsNone(alerts.first().resolved_at)

    def test_low_stock_cooldown_no_duplicate_alert(self):
        part = self.make_part(on_hand=10, reorder_point=5)
        client = APIClient()
        client.credentials(**auth_header(self.engineer))

        # First movement drops it below reorder_point.
        client.post(
            "/api/v1/stock-movements/",
            {"spare_part": str(part.id), "movement_type": "ADJUSTMENT", "quantity": -6, "reason": "count"},
            format="json",
        )
        first_alert = Alert.objects.get(alert_type=Alert.AlertType.LOW_STOCK, spare_part_id=part.id)
        first_seen = first_alert.last_seen_at

        # Second movement still below reorder_point — must update
        # last_seen_at on the SAME alert, not create a second one.
        client.post(
            "/api/v1/stock-movements/",
            {"spare_part": str(part.id), "movement_type": "ADJUSTMENT", "quantity": -1, "reason": "count 2"},
            format="json",
        )
        alerts = Alert.objects.filter(alert_type=Alert.AlertType.LOW_STOCK, spare_part_id=part.id)
        self.assertEqual(alerts.count(), 1)
        alerts.first().refresh_from_db()
        self.assertGreaterEqual(alerts.first().last_seen_at, first_seen)

    def test_receipt_crossing_reorder_point_resolves_low_stock(self):
        part = self.make_part(on_hand=10, reorder_point=5)
        # Seed an open LOW_STOCK alert directly (as if a prior movement
        # already dropped it below reorder_point).
        now = timezone.now()
        Alert.objects.create(
            bearing=None,
            spare_part=part,
            alert_type=Alert.AlertType.LOW_STOCK,
            severity=Alert.Severity.WARNING,
            triggered_at=now,
            last_seen_at=now,
        )
        part.quantity_on_hand = 2
        part.save(update_fields=["quantity_on_hand"])

        client = APIClient()
        client.credentials(**auth_header(self.engineer))
        resp = client.post(
            "/api/v1/stock-movements/",
            {"spare_part": str(part.id), "movement_type": "RECEIPT", "quantity": 10},
            format="json",
        )
        self.assertEqual(resp.status_code, 201, resp.data)
        part.refresh_from_db()
        self.assertEqual(part.quantity_on_hand, 12)

        alert = Alert.objects.get(alert_type=Alert.AlertType.LOW_STOCK, spare_part_id=part.id)
        self.assertIsNotNone(alert.resolved_at)
        self.assertIsNone(alert.resolved_by)

    def test_receipt_creates_stock_movement_row(self):
        part = self.make_part(on_hand=5)
        client = APIClient()
        client.credentials(**auth_header(self.admin))
        resp = client.post(
            "/api/v1/stock-movements/",
            {"spare_part": str(part.id), "movement_type": "RECEIPT", "quantity": 4},
            format="json",
        )
        self.assertEqual(resp.status_code, 201, resp.data)
        movement = StockMovement.objects.get(spare_part=part)
        self.assertEqual(movement.movement_type, StockMovement.MovementType.RECEIPT)
        self.assertEqual(movement.quantity, 4)
        self.assertIsNone(movement.reference_work_order)
