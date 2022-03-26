from pyexpat import model
from django.db import models

# In django models are python representation of SQL table
# We will keep all the user-drawn images and their correct label
# In order to accumulate more data and hopefully make a more accurate CNN
class UserDrawnImage(models.Model):
    LABEL_CHOICES = [
        (0, '0'),
        (1, '1'),
        (2, '2'),
        (3, '3'),
        (4, '4'),
        (5, '5'),
        (6, '6'),
        (7, '7'),
        (8, '8'),
        (9, '9'),
    ]

    # After we ask from the user a feedback - the label is set to moderation
    # Due to the fact that we want data to be accurate. We of course cannot expect
    # our users to be honest :) Thus we will actually have to filter out manually 
    # returned answers that are incorrect

    STATUS_MODERATION = 0
    STATUS_CONFIRMED = 1

    STATUS_CHOICES = [
        (STATUS_MODERATION, 'Moderation'),
        (STATUS_CONFIRMED, 'Confirmed')
    ]

    label = models.CharField(choices=LABEL_CHOICES, null=True, max_length=1)
    status = models.CharField(choices=STATUS_CHOICES, default=STATUS_MODERATION, max_length=255)
    image = models.ImageField()