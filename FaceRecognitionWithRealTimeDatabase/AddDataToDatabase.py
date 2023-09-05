import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendacerealtime-800cd-default-rtdb.firebaseio.com/"
})

ref = db.reference('Students')

data = {
    "groop": {
        "ИСТ": {
            "108481": {
                "name": "qwer",
                "major": "Informatics",
                "starting_year": 2020,
                "total_attendance": 4,
                "standing": "ИСТ",
                "year": 3,
                "last_attendance_time": "2022-12-11 00:54:34"
            },
        },
        "ПКСТ": {
            "108481": {
                "name": "Beketov Vitaliy",
                "major": "Informatics",
                "starting_year": 2020,
                "total_attendance": 4,
                "standing": "ПКСТ",
                "year": 3,
                "last_attendance_time": "2022-12-11 00:54:34"
            },
            "321654": {
                "name": "Murtaza Hassan",
                "major": "Robotics",
                "starting_year": 2017,
                "total_attendance": 7,
                "standing": "G",
                "year": 4,
                "last_attendance_time": "2022-12-11 00:54:34"
            },
            "852741": {
                "name": "Emly Blunt",
                "major": "Economics",
                "starting_year": 2021,
                "total_attendance": 12,
                "standing": "B",
                "year": 1,
                "last_attendance_time": "2022-12-11 00:54:34"
            },
            "963852": {
                "name": "Elon Musk",
                "major": "Physics",
                "starting_year": 2020,
                "total_attendance": 7,
                "standing": "G",
                "year": 2,
                "last_attendance_time": "2022-12-11 00:54:34"
            }
        },
    }
}

# JSON in Python is Dictionary
for key, value in data.items():
    ref.child(key).set(value)