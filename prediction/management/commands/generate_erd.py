import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Dissertation.settings')
django.setup()

from django_er_diagram.erd import generate_erd

# Generate the ERD
generate_erd(
    apps=['data', 'dashboard', 'prediction'],
    output_file='erd.png',
    format='png'
)