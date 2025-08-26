from django.contrib import admin
from .models import RaceIncident

@admin.register(RaceIncident)
class RaceIncidentAdmin(admin.ModelAdmin):
    list_display = ('year', 'round', 'event_name', 'type', 'driver_name', 'lap', 'created_at')
    list_filter = ('year', 'type')
    search_fields = ('event_name', 'driver_name', 'description')
    ordering = ('-year', 'round', 'lap')
