from django.contrib.auth import get_user_model, password_validation
from rest_framework import serializers

User = get_user_model()


def other_active_admin_exists(user):
    return (
        User.objects.filter(role=User.Role.ADMIN, is_active=True).exclude(pk=user.pk).exists()
    )


class UserSerializer(serializers.ModelSerializer):
    """
    api.md: Users — read/write ADMIN. `password` write-only (hash tidak pernah
    keluar). `is_staff`/`is_superuser` sengaja tidak diekspos: hak akses Django
    admin terpisah dari `role` aplikasi.
    """

    password = serializers.CharField(
        write_only=True, required=False, trim_whitespace=False, style={"input_type": "password"}
    )

    class Meta:
        model = User
        fields = [
            "id",
            "username",
            "email",
            "first_name",
            "last_name",
            "role",
            "is_active",
            "date_joined",
            "password",
        ]
        read_only_fields = ["id", "date_joined"]

    def validate_password(self, value):
        password_validation.validate_password(value, user=self.instance)
        return value

    def validate(self, attrs):
        inst = self.instance
        if inst is None:
            if not attrs.get("password"):
                raise serializers.ValidationError({"password": "Wajib diisi saat membuat user."})
            return attrs
        new_role = attrs.get("role", inst.role)
        new_active = attrs.get("is_active", inst.is_active)
        if (
            inst.role == User.Role.ADMIN
            and inst.is_active
            and (new_role != User.Role.ADMIN or not new_active)
            and not other_active_admin_exists(inst)
        ):
            raise serializers.ValidationError(
                "Tidak bisa menurunkan atau menonaktifkan ADMIN aktif terakhir."
            )
        return attrs

    def create(self, validated_data):
        password = validated_data.pop("password")
        return User.objects.create_user(password=password, **validated_data)

    def update(self, instance, validated_data):
        password = validated_data.pop("password", None)
        instance = super().update(instance, validated_data)
        if password:
            instance.set_password(password)
            instance.save(update_fields=["password"])
        return instance
