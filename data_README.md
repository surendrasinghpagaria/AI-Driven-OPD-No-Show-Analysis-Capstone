# Dataset

## Source
**Kaggle — Medical Appointment No Shows**
https://www.kaggle.com/datasets/joniarroba/noshowappointments

## Download Instructions
1. Visit the Kaggle link above (free account required)
2. Click **Download** → download `archive.zip`
3. Extract `KaggleV2-May-2016.csv`
4. Place the CSV file in the **root folder** of this repository (same level as the notebook)

## File Details
- **Filename:** `KaggleV2-May-2016.csv`
- **Size:** 10.2 MB
- **Records:** 110,527 rows (110,522 after removing 5 invalid age records)
- **Columns:** 14

## Column Descriptions

| Column | Type | Description |
|---|---|---|
| PatientId | Float | Unique patient identifier |
| AppointmentID | Integer | Unique appointment identifier |
| Gender | String | Patient gender: F (Female) or M (Male) |
| ScheduledDay | DateTime | Date and time appointment was booked |
| AppointmentDay | DateTime | Date of the actual appointment |
| Age | Integer | Patient age in years (0–115) |
| Neighbourhood | String | Neighbourhood in Vitória, Brazil (81 unique values) |
| Scholarship | Binary | 1 = enrolled in Bolsa Família welfare programme |
| Hipertension | Binary | 1 = patient has hypertension |
| Diabetes | Binary | 1 = patient has diabetes |
| Alcoholism | Binary | 1 = patient has alcoholism |
| Handcap | Integer | Disability level (0–4) |
| SMS_received | Binary | 1 = patient received at least one SMS reminder |
| No-show | String | Target variable: "Yes" = patient did NOT attend |

## Engineered Features (created in notebook)
| Feature | Description |
|---|---|
| WaitDays | AppointmentDay − ScheduledDay (days), clipped at 0 |
| WaitBucket | Categorical: 0, 1–7, 8–15, 16–30, 31+ days |
| NoShow | Binary encoding of No-show column (Yes=1, No=0) |
| Gender_enc | Binary encoding of Gender (M=1, F=0) |
| Nbhd_enc | Label-encoded Neighbourhood |
