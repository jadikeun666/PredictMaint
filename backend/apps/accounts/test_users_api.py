"""API /users/ (ADMIN) + /users/me/."""
from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from apps.exports.models import ExportJob

User = get_user_model()
URL = "/api/v1/users/"
PW = "S3cret-pass-123"


def mk(username, role, **kw):
    return User.objects.create_user(username=username, password=PW, role=role, **kw)


class UserApiTests(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.admin = mk("u_admin", User.Role.ADMIN)
        cls.engineer = mk("u_eng", User.Role.ENGINEER)
        cls.technician = mk("u_tech", User.Role.TECHNICIAN)
        cls.viewer = mk("u_view", User.Role.VIEWER)

    # ---- akses
    def test_unauthenticated_401(self):
        self.assertEqual(self.client.get(URL).status_code, 401)
        self.assertEqual(self.client.get(f"{URL}me/").status_code, 401)

    def test_non_admin_forbidden_everywhere(self):
        for u in (self.engineer, self.technician, self.viewer):
            self.client.force_authenticate(u)
            self.assertEqual(self.client.get(URL).status_code, 403, u.role)
            self.assertEqual(self.client.get(f"{URL}{self.admin.id}/").status_code, 403)
            self.assertEqual(self.client.post(URL, {}, format="json").status_code, 403)
            self.assertEqual(self.client.patch(f"{URL}{self.viewer.id}/", {"role": "ADMIN"}, format="json").status_code, 403)
            self.assertEqual(self.client.delete(f"{URL}{self.viewer.id}/").status_code, 403)
        self.viewer.refresh_from_db()
        self.assertEqual(self.viewer.role, "VIEWER")

    def test_me_for_every_role_without_secrets(self):
        for u in (self.admin, self.engineer, self.technician, self.viewer):
            self.client.force_authenticate(u)
            r = self.client.get(f"{URL}me/")
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertEqual(body["username"], u.username)
            self.assertEqual(body["role"], u.role)
            self.assertNotIn("password", body)

    # ---- list / retrieve
    def test_admin_list_sorted_without_password_or_staff_flags(self):
        self.client.force_authenticate(self.admin)
        r = self.client.get(URL)
        self.assertEqual(r.status_code, 200)
        names = [x["username"] for x in r.json()["results"]]
        self.assertEqual(names, sorted(names))
        for row in r.json()["results"]:
            self.assertNotIn("password", row)
            self.assertNotIn("is_staff", row)
            self.assertNotIn("is_superuser", row)
        self.assertEqual(self.client.get(f"{URL}{self.viewer.id}/").status_code, 200)

    def test_filters_and_invalid_filters(self):
        self.client.force_authenticate(self.admin)
        r = self.client.get(URL, {"role": "VIEWER"})
        self.assertEqual([x["username"] for x in r.json()["results"]], ["u_view"])
        self.viewer.is_active = False
        self.viewer.save()
        r = self.client.get(URL, {"is_active": "false"})
        self.assertEqual([x["username"] for x in r.json()["results"]], ["u_view"])
        for p in ({"role": "BOSS"}, {"is_active": "maybe"}):
            r = self.client.get(URL, p)
            self.assertEqual(r.status_code, 400)
            self.assertEqual(r.json()["code"], "invalid_query_param")

    # ---- create
    def test_create_requires_password_and_hides_it(self):
        self.client.force_authenticate(self.admin)
        r = self.client.post(URL, {"username": "baru", "role": "VIEWER"}, format="json")
        self.assertEqual(r.status_code, 400)
        self.assertIn("password", r.json()["errors"])
        r = self.client.post(URL, {"username": "baru", "role": "TECHNICIAN", "password": PW}, format="json")
        self.assertEqual(r.status_code, 201, r.content)
        self.assertNotIn("password", r.json())
        self.client.force_authenticate(None)
        t = self.client.post("/api/v1/auth/token/", {"username": "baru", "password": PW}, format="json")
        self.assertEqual(t.status_code, 200)
        self.assertIn("access", t.json())
        self.assertNotEqual(User.objects.get(username="baru").password, PW)  # tersimpan sebagai hash

    def test_create_duplicate_username_and_bad_role_400(self):
        self.client.force_authenticate(self.admin)
        r = self.client.post(URL, {"username": "u_view", "role": "VIEWER", "password": PW}, format="json")
        self.assertEqual(r.status_code, 400)
        self.assertIn("username", r.json()["errors"])
        r = self.client.post(URL, {"username": "x1", "role": "BOSS", "password": PW}, format="json")
        self.assertEqual(r.status_code, 400)
        self.assertIn("role", r.json()["errors"])

    # ---- update
    def test_patch_role_and_password(self):
        self.client.force_authenticate(self.admin)
        r = self.client.patch(f"{URL}{self.viewer.id}/", {"role": "ENGINEER"}, format="json")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["role"], "ENGINEER")
        r = self.client.patch(f"{URL}{self.viewer.id}/", {"password": "Baru-pass-456!"}, format="json")
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("password", r.json())
        self.client.force_authenticate(None)
        ok = self.client.post("/api/v1/auth/token/", {"username": "u_view", "password": "Baru-pass-456!"}, format="json")
        old = self.client.post("/api/v1/auth/token/", {"username": "u_view", "password": PW}, format="json")
        self.assertEqual((ok.status_code, old.status_code), (200, 401))

    def test_cannot_demote_or_deactivate_last_active_admin(self):
        self.client.force_authenticate(self.admin)
        for payload in ({"role": "ENGINEER"}, {"is_active": False}):
            r = self.client.patch(f"{URL}{self.admin.id}/", payload, format="json")
            self.assertEqual(r.status_code, 400, payload)
            self.assertEqual(r.json()["code"], "validation_error")
        self.admin.refresh_from_db()
        self.assertEqual(self.admin.role, "ADMIN")
        self.assertTrue(self.admin.is_active)

    def test_can_demote_admin_when_another_active_admin_exists(self):
        admin2 = mk("u_admin2", User.Role.ADMIN)
        self.client.force_authenticate(self.admin)
        r = self.client.patch(f"{URL}{admin2.id}/", {"role": "VIEWER"}, format="json")
        self.assertEqual(r.status_code, 200)

    # ---- delete
    def test_delete_self_409_conflict(self):
        self.client.force_authenticate(self.admin)
        r = self.client.delete(f"{URL}{self.admin.id}/")
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["code"], "conflict")
        self.assertTrue(User.objects.filter(pk=self.admin.pk).exists())

    def test_delete_other_user_204(self):
        self.client.force_authenticate(self.admin)
        self.assertEqual(self.client.delete(f"{URL}{self.technician.id}/").status_code, 204)
        self.assertFalse(User.objects.filter(pk=self.technician.pk).exists())

    def test_delete_referenced_user_409_protected(self):
        ExportJob.objects.create(export_type="HISTORY_EXCEL", requested_by=self.engineer, params={})
        self.client.force_authenticate(self.admin)
        r = self.client.delete(f"{URL}{self.engineer.id}/")
        self.assertEqual(r.status_code, 409)
        body = r.json()
        for k in ("type", "title", "status", "detail"):
            self.assertIn(k, body)
        self.assertIn("exports.ExportJob", body["detail"])
