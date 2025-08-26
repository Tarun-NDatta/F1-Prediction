from django.db import models

class RaceIncident(models.Model):
    year = models.IntegerField(default=2025)
    round = models.IntegerField()
    event_name = models.CharField(max_length=200)
    lap = models.IntegerField(null=True, blank=True)
    driver_name = models.CharField(max_length=100, null=True, blank=True)
    type = models.CharField(
        max_length=100,
        choices=[
            ('PENALTY', 'Penalty'),
            ('SAFETY_CAR', 'Safety Car'),
            ('VSC', 'Virtual Safety Car'),
            ('DNF', 'Did Not Finish'),
            ('START_ISSUE', 'Start Issue'),
            ('PIT_STOP', 'Pit Stop'),
            ('WEATHER', 'Weather'),
            ('OTHER', 'Other'),
        ],
        default='OTHER'
    )
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['year', 'round', 'lap', 'driver_name']

    def __str__(self):
        return f"{self.year} R{self.round} - {self.event_name}: {self.type} ({self.driver_name or 'N/A'})"
