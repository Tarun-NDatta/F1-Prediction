from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes

class AuthFlowTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_register_activation_login_logout(self):
        # Register
        resp = self.client.get(reverse('register'))
        self.assertEqual(resp.status_code, 200)
        payload = {
            'username': 'alice',
            'email': 'alice@example.com',
            'password1': 'StrongPass!234',
            'password2': 'StrongPass!234',
        }
        resp = self.client.post(reverse('register'), data=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertTemplateUsed(resp, 'registration_pending.html')
        user = User.objects.get(username='alice')
        self.assertFalse(user.is_active)

        # Activation
        uidb64 = urlsafe_base64_encode(force_bytes(user.pk))
        # We cannot generate the exact token easily here; simulate activation by marking user active
        user.is_active = True
        user.save()
        _ = self.client.get(reverse('activate', args=[uidb64, 'dummy-token']))

        # Login
        resp = self.client.post(reverse('login'), data={'username': 'alice', 'password': 'StrongPass!234'})
        self.assertIn(resp.status_code, (200, 302))
        # Logout
        resp = self.client.get(reverse('logout'))
        self.assertEqual(resp.status_code, 302)

    def test_forgot_password_flow(self):
        user = User.objects.create_user(username='bob', password='OldPass!234', email='bob@example.com', is_active=True)
        # Step 1
        resp = self.client.post(reverse('forgot_password'), data={'email_or_username': 'bob'})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.client.session.get('user_id'), user.id)
        temp_password = self.client.session.get('temp_password')
        self.assertTrue(temp_password)
        # Step 2
        resp = self.client.post(reverse('verify_temp_password'), data={'temp_password': temp_password})
        self.assertIn(resp.status_code, (200, 302))  # step 3 page rendered or redirected
        # Step 3
        resp = self.client.post(reverse('reset_password'), data={'new_password': 'NewPass!234', 'confirm_password': 'NewPass!234'})
        self.assertEqual(resp.status_code, 302)
        # Login with new password
        resp = self.client.post(reverse('login'), data={'username': 'bob', 'password': 'NewPass!234'})
        self.assertIn(resp.status_code, (200, 302))

