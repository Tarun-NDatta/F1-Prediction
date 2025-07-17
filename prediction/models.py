from django.db import models
from data.models import Driver, Event 

class ridgeregression(models.Model):
    driver = models.ForeignKey(Driver, on_delete=models.CASCADE)
    event = models.ForeignKey(Event, on_delete=models.CASCADE)
    
    year = models.IntegerField()
    round_number = models.IntegerField()

    predicted_position = models.FloatField()
    actual_position = models.IntegerField(null=True, blank=True)

    model_name = models.CharField(max_length=100, default='ridge_regression')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('driver', 'event', 'model_name')
        indexes = [
            models.Index(fields=['year', 'round_number', 'model_name']),
        ]
        ordering = ['predicted_position']

    def __str__(self):
        return f"{self.model_name.upper()} | {self.driver} | {self.event} → Predicted: {self.predicted_position:.2f}"
