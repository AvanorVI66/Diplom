import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
import shutil
import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone
from datetime import datetime

import tkinter.filedialog as fd
import shutil

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendacerealtime-800cd-default-rtdb.firebaseio.com/",
    'storageBucket': "faceattendacerealtime-800cd.appspot.com"
})
ref = db.reference('Students')
folderPath = 'Images'
image_path = ''  # Добавленная переменная
bucket = storage.bucket()

def run_main_program():
    cap = cv2.VideoCapture(0)

    # Image Capture Dimensions
    cap.set(3, 640)
    cap.set(4, 480)

    imgBackground = cv2.imread('Resources/background.png')

    # Importing the mode images into the list
    folderModePath = 'Resources/Modes'
    modePathList = os.listdir(folderModePath)
    imgModeList = []
    for path in modePathList:
        imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

    # print(len(imgModeList))

    # Load the encoding files
    file = open('EncodeFile.p', 'rb')
    encodeListKnownWithIds = pickle.load(file)
    file.close()
    encodeListKnown, studentIds = encodeListKnownWithIds
    print(studentIds)

    modeType = 0
    counter = 0
    id = -1
    imgStudent = []

    while True:
        success, img = cap.read()

        # Minimizing the computation power by making the images smaller only for face recognition, not the webcam frame
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        # create the webcam video inside the imgBackground / Mask object
        # 162, 162+480 = height start & end point, 55, 55+640 = width start & end point
        imgBackground[162:162 + 480, 55:55 + 640] = img

        # create the mode images inside the imgBackground / Mask object
        # 44, 44+633 = height start & end point, 808, 808+414 = width start & end point
        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

        # if face is detected
        if faceCurFrame:
            # Looping 2 lists at once using zip method
            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)

                # the lower faceDis, the bigger possibilities that 2 images are same
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                # print("matches", matches)
                # print("faceDis", faceDis)

                matchIndex = np.argmin(faceDis)
                # print("MatchIndex", matchIndex)

                if matches[matchIndex]:
                    # print("Known Face Detected")
                    # print(studentIds[matchIndex])

                    # Create a bounding box start from the webcam frame
                    y1, x2, y2, x1 = faceLoc

                    # Multiplied 4 because of the previous resizing
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1

                    # Bounding box in cvzone are more vancy than openCV
                    imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                    id = studentIds[matchIndex]

                    if counter == 0:
                        cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                        cv2.imshow("Face Attendance", imgBackground)
                        cv2.waitKey(1)
                        counter = 1
                        modeType = 1

            if counter != 0:
                if counter == 1:
                    # Get the Data from firebase db
                    studentInfo = db.reference(f'Students/groop/ПКСТ/{id}').get()
                    print(studentInfo)

                    # Get the Image from firebase storage
                    blob = bucket.get_blob(f'Images/{id}.png')

                    # download image as a string and decode it
                    array = np.frombuffer(blob.download_as_string(), np.uint8)
                    imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

                    # Update data of attendance
                    datetimeObject = datetime.strptime(studentInfo['last_attendance_time'],
                                                       "%Y-%m-%d %H:%M:%S")
                    secondsElapsed = (datetime.now() - datetimeObject).total_seconds()  # now - timeLastAttendance
                    print(secondsElapsed)

                    # If now - timeLastAttendance is greater than 30 seconds
                    if secondsElapsed > 30:
                        ref = db.reference(f'Students/groop/ПКСТ/{id}')
                        studentInfo['total_attendance'] += 1
                        ref.child('total_attendance').set(studentInfo['total_attendance'])
                        ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    else:
                        modeType = 3
                        counter = 0
                        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                if modeType != 3:

                    if 10 < counter < 20:
                        modeType = 2

                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                    if counter <= 10:
                        # Put the text from firebase db in the Image Background
                        cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(id), (1006, 493),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(studentInfo['standing']), (910, 625),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                        cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                        cv2.putText(imgBackground, str(studentInfo['starting_year']), (1125, 625),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                        # Put the text from firebase db centered in the Image Background
                        (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                        offset = (414 - w) // 2  # left distance to center
                        cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                        # Put the image student in the image background
                        imgBackground[175:175 + 216, 909:909 + 216] = imgStudent

                    counter += 1

                    if counter >= 20:
                        counter = 0
                        modeType = 0
                        studentInfo = []
                        imgStudent = []
                        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
        else:
            modeType = 0
            counter = 0

        # cv2.imshow("Webcam", img) # show window of your camera
        cv2.imshow("Face Attendance", imgBackground)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q To exit
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Запуск программы", "Запущена главная программа машинного зрения")

def run_student_program():
    def upload_image_to_storage(image_path):
        # Здесь должна быть ваша логика для загрузки фотографии в Firebase Storage
        pass

    def add_student_to_firebase():
        global image_path

        # Получение значений из текстовых полей
        student_id = student_id_entry.get()
        name = name_entry.get()
        major = major_entry.get()
        starting_year = starting_year_entry.get()
        total_attendance = total_attendance_entry.get()
        standing = standing_entry.get()
        year = year_entry.get()
        group = group_combobox.get()  # Получение выбранной группы

        if not image_path:
            messagebox.showwarning("Ошибка", "Пожалуйста, выберите фотографию студента.")
            return

        # Копирование фотографии студента в директорию 'Images'
        image_filename = os.path.basename(image_path)
        destination_path = os.path.join(folderPath, image_filename)
        shutil.copyfile(image_path, destination_path)

        # Загрузка фотографии студента в Firebase Storage
        upload_image_to_storage(destination_path)

        # Формирование данных студента
        student_data = {
            "name": name,
            "major": major,
            "starting_year": starting_year,
            "total_attendance": total_attendance,
            "standing": standing,
            "year": year,
            "last_attendance_time": "0000-00-00 00:00:00"
        }

        # Добавление данных студента в базу данных Firebase
        group_ref = ref.child('groop').child(group)  # Получение ссылки на узел группы
        new_student_ref = group_ref.child(student_id)  # Получение ссылки на узел нового студента
        new_student_ref.set(student_data)

        # Очистка текстовых полей
        student_id_entry.delete(0, tk.END)
        name_entry.delete(0, tk.END)
        major_entry.delete(0, tk.END)
        starting_year_entry.delete(0, tk.END)
        total_attendance_entry.delete(0, tk.END)
        standing_entry.delete(0, tk.END)
        year_entry.delete(0, tk.END)

        messagebox.showinfo("Добавление студента", "Данные студента успешно добавлены в базу данных Firebase")

    def browse_image():
        global image_path
        image_path = filedialog.askopenfilename(initialdir="/", title="Выберите фотографию студента", filetypes=(("PNG files", "*.png"),))
        # Можете добавить код для отображения выбранной фотографии или её имени на графическом интерфейсе

    # Создание графического интерфейса для программы добавления студентов
    window = tk.Toplevel()
    window.title("Программа добавления студентов")

    group_label = tk.Label(window, text="Группа:")
    group_label.pack()

    group_options = ["ИСТ", "ПКСТ"]  # Здесь можно добавить другие варианты групп
    group_combobox = ttk.Combobox(window, values=group_options)
    group_combobox.pack()

    student_id_label = tk.Label(window, text="ID студента:")
    student_id_label.pack()
    student_id_entry = tk.Entry(window)
    student_id_entry.pack()

    name_label = tk.Label(window, text="Имя:")
    name_label.pack()
    name_entry = tk.Entry(window)
    name_entry.pack()

    major_label = tk.Label(window, text="Специальность:")
    major_label.pack()
    major_entry = tk.Entry(window)
    major_entry.pack()

    starting_year_label = tk.Label(window, text="Год поступления:")
    starting_year_label.pack()
    starting_year_entry = tk.Entry(window)
    starting_year_entry.pack()

    total_attendance_label = tk.Label(window, text="Общая посещаемость:")
    total_attendance_label.pack()
    total_attendance_entry = tk.Entry(window)
    total_attendance_entry.pack()

    standing_label = tk.Label(window, text="Группа:")
    standing_label.pack()
    standing_entry = tk.Entry(window)
    standing_entry.pack()

    year_label = tk.Label(window, text="Курс:")
    year_label.pack()
    year_entry = tk.Entry(window)
    year_entry.pack()

    browse_button = tk.Button(window, text="Выбрать фотографию", command=browse_image)
    browse_button.pack()

    add_button = tk.Button(window, text="Добавить студента", command=add_student_to_firebase)
    add_button.pack(pady=10)

def edit_student():
    def update_student():
        selected_group = group_combobox.get()  # Получение выбранной группы
        student_id = student_id_entry.get()  # Получение ID студента для редактирования

        if selected_group == "" or student_id == "":
            messagebox.showwarning("Ошибка", "Пожалуйста, выберите группу и введите ID студента.")
            return

        group_ref = ref.child('groop').child(selected_group)  # Получение ссылки на узел группы
        student_ref = group_ref.child(student_id)  # Получение ссылки на узел студента

        if student_ref.get() is None:
            messagebox.showwarning("Ошибка", "Студент с указанным ID не найден.")
        else:
            # Получение данных студента
            student_data = student_ref.get()
            name = student_data.get("name", "")
            major = student_data.get("major", "")
            starting_year = student_data.get("starting_year", "")
            total_attendance = student_data.get("total_attendance", "")
            standing = student_data.get("standing", "")
            year = student_data.get("year", "")
            last_attendance_time = student_data.get("last_attendance_time", "")

            # Создание окна редактирования студента
            edit_window = tk.Toplevel()
            edit_window.title("Редактирование студента")

            name_label = tk.Label(edit_window, text="Имя:")
            name_label.pack()
            name_entry = tk.Entry(edit_window)
            name_entry.pack()
            name_entry.insert(0, name)  # Заполнение поля имени

            major_label = tk.Label(edit_window, text="Специальность:")
            major_label.pack()
            major_entry = tk.Entry(edit_window)
            major_entry.pack()
            major_entry.insert(0, major)  # Заполнение поля специальности

            starting_year_label = tk.Label(edit_window, text="Год поступления:")
            starting_year_label.pack()
            starting_year_entry = tk.Entry(edit_window)
            starting_year_entry.pack()
            starting_year_entry.insert(0, starting_year)  # Заполнение поля года поступления

            total_attendance_label = tk.Label(edit_window, text="Общее посещение:")
            total_attendance_label.pack()
            total_attendance_entry = tk.Entry(edit_window)
            total_attendance_entry.pack()
            total_attendance_entry.insert(0, total_attendance)  # Заполнение поля общего посещения

            standing_label = tk.Label(edit_window, text="Группа:")
            standing_label.pack()
            standing_entry = tk.Entry(edit_window)
            standing_entry.pack()
            standing_entry.insert(0, standing)  # Заполнение поля уровня успеваемости

            year_label = tk.Label(edit_window, text="Курс:")
            year_label.pack()
            year_entry = tk.Entry(edit_window)
            year_entry.pack()
            year_entry.insert(0, year)  # Заполнение поля курса

            last_attendance_time_label = tk.Label(edit_window, text="Последнее посещение:")
            last_attendance_time_label.pack()
            last_attendance_time_entry = tk.Entry(edit_window)
            last_attendance_time_entry.pack()
            last_attendance_time_entry.insert(0, last_attendance_time)  # Заполнение поля последнего посещения

            def save_changes():
                # Обновление данных студента
                student_data = {
                    "name": name_entry.get(),
                    "major": major_entry.get(),
                    "starting_year": starting_year_entry.get(),
                    "total_attendance": total_attendance_entry.get(),
                    "standing": standing_entry.get(),
                    "year": year_entry.get(),
                    "last_attendance_time": last_attendance_time_entry.get()
                }
                student_ref.update(student_data)
                messagebox.showinfo("Редактирование студента", "Данные студента успешно обновлены.")
                edit_window.destroy()

            save_button = tk.Button(edit_window, text="Сохранить", command=save_changes)
            save_button.pack(pady=10)

    def confirm_delete():
        group = group_combobox.get()  # Получение выбранной группы
        student_id = student_id_entry.get()  # Получение ID студента для удаления

        if group == "" or student_id == "":
            messagebox.showwarning("Ошибка", "Пожалуйста, выберите группу и введите ID студента.")
            return

        group_ref = ref.child('groop').child(group)  # Получение ссылки на узел группы
        student_ref = group_ref.child(student_id)  # Получение ссылки на узел студента

        if student_ref.get() is None:
            messagebox.showwarning("Ошибка", "Студент с указанным ID не найден.")
        else:
            student_ref.delete()
            messagebox.showinfo("Удаление студента", "Студент успешно удален из базы данных.")


    # Создание окна редактирования/удаления студента
    edit_window = tk.Toplevel()
    edit_window.title("Редактирование/удаление студента")

    group_label = tk.Label(edit_window, text="Группа:")
    group_label.pack()

    group_options = ["ИСТ", "ПКСТ"]  # Добавьте другие варианты групп здесь
    group_combobox = ttk.Combobox(edit_window, values=group_options)
    group_combobox.pack()

    student_id_label = tk.Label(edit_window, text="ID студента:")
    student_id_label.pack()
    student_id_entry = tk.Entry(edit_window)
    student_id_entry.pack()

    edit_button = tk.Button(edit_window, text="Редактировать", command=update_student)
    edit_button.pack(pady=10)

    delete_button = tk.Button(edit_window, text="Удалить", command=confirm_delete)
    delete_button.pack(pady=5)


def show_attendance_info():
    # Function to display attendance information for students
    def display_attendance_info():
        selected_group = group_combobox.get()  # Get the selected group

        if selected_group == "":
            messagebox.showwarning("Ошибка", "Пожалуйста, выберите группу.")
            return

        group_ref = ref.child('groop').child(selected_group)  # Reference to the selected group

        # Retrieve student information for the selected group from the Firebase database
        students = group_ref.get()

        # Create a text widget to display attendance information
        text_widget = tk.Text(attendance_window)
        text_widget.pack()

        if students is None:
            text_widget.insert(tk.END, "Нет информации о посещаемости для выбранной группы.")
        else:
            # Iterate over each student and display their attendance information
            for student_id, student_info in students.items():
                attendance_text = f"ID: {student_id}\n" \
                                  f"Имя: {student_info['name']}\n" \
                                  f"Общая посещаемость: {student_info['total_attendance']}\n" \
                                  f"Последнее посещение: {student_info['last_attendance_time']}\n\n"

                text_widget.insert(tk.END, attendance_text)

    attendance_window = tk.Toplevel()
    attendance_window.title("Информация о посещаемости студентов")

    group_label = tk.Label(attendance_window, text="Группа:")
    group_label.pack()

    group_options = ["ИСТ", "ПКСТ"]  # Здесь можно добавить другие варианты групп
    group_combobox = ttk.Combobox(attendance_window, values=group_options)
    group_combobox.pack()

    display_button = tk.Button(attendance_window, text="Показать информацию", command=display_attendance_info)
    display_button.pack(pady=10)

def main():
    # Создание главного окна
    root = tk.Tk()
    root.title("Выберите программу")

    # Создание кнопок выбора программы
    main_button = tk.Button(root, text="Запустить программу машинного зрения", command=run_main_program)
    main_button.pack(pady=10)

    student_button = tk.Button(root, text="Запустить программу добавления студентов", command=run_student_program)
    student_button.pack(pady=10)

    edit_student_button = tk.Button(root, text="Редактирование студентов", command=edit_student)
    edit_student_button.pack(pady=10)

    info_button = tk.Button(root, text="Запустить программу статистики", command=show_attendance_info)
    info_button.pack(pady=10)

    # Запуск основного цикла событий
    root.mainloop()

if __name__ == "__main__":
    main()