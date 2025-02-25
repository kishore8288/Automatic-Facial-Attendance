# Automatic-Facial-Attendance
**Steps of my approach :**
`1. Face extraction from the whole image`
`2. Building a Siamese Network from scratch`
`3. Giving this image to siamese model`
`4. Extracting the person with high score with image`
`5. Updating attendance to that person dynamically`

#### Reason for using Siamese model : 
    This particular model comes under the few-shot learning. 
    since instead of understanding and memorizing the data flow, 
    it effective makes relation betweent the existing database and the test image. 
    This can be done using few images and this model is widely used in the fingerprint recognition in those days.
