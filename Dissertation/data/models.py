from django.db import models

class Circuit(models.Model):
    name = models.CharField(max_length=100)
    location = models.CharField(max_length=100)
    country = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class Race(models.Model):
    year = models.IntegerField()
    round = models.IntegerField()
    name = models.CharField(max_length=100)
    date = models.DateField()
    circuit = models.ForeignKey(Circuit, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.year} {self.name} Round {self.round}"

class Team(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class Driver(models.Model):
    driver_id = models.CharField(max_length=20, unique=True)
    given_name = models.CharField(max_length=50)
    family_name = models.CharField(max_length=50)
    nationality = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.given_name} {self.family_name}"

class Result(models.Model):
    race = models.ForeignKey(Race, on_delete=models.CASCADE)
    driver = models.ForeignKey(Driver, on_delete=models.CASCADE)
    team = models.ForeignKey(Team, on_delete=models.CASCADE)
    position = models.IntegerField(null=True)
    points = models.FloatField(null=True)
    laps = models.IntegerField(null=True)
    status = models.CharField(max_length=50, null=True)
    time = models.CharField(max_length=50, null=True)  # race time or gap

    def __str__(self):
        return f"{self.race} - {self.driver} - Pos {self.position}"
