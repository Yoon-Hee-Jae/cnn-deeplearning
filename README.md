#  웹캠을 통한 감정 분석 프로젝트

# 목차
- 개요
- 사용 데이터
- 모델 구성

  0. 데이터 준비
  1. 감정분석 모델
  2. 나이판별 모델
  3. 성별판별 모델
  4. 웹캠 적용
 
- 결과물물

# 1. 개요
노트북에 탑재되어 있는 웹캠을 활용하여 웹캠에 비친 사용자의 얼굴을 분석하여 성별, 나이 그리고 감정을 분석할 수 있는 기능을 구현해보았습니다.

# 2. 사용 데이터
## 1. facial expression recognition competition dataset (filename = 'fer2013.csv)
   kaggle에 있는 데이터로서 각 이미지마다 감정이 라벨링되어 있는 데이터셋입니다.
   
<예시 이미지>

![image](https://github.com/Yoon-Hee-Jae/cnn-deeplearning/assets/140389762/57dc3371-fc1f-4237-8db4-9f041275d860)


## 2. UTK face dataset
   다양한 연령대 및 인종에 대한 이미지 데이터셋입니다. 
   각 이미지는 파일명에 연령, 성별 그리고 인종에 대한 정보를 가지고 있습니다.
   따라서 사용시 파일명에 따라 각 이미지를 라벨링해주는 작업을 해주어야만 합니다.

<예시 이미지>

![image](https://github.com/Yoon-Hee-Jae/cnn-deeplearning/assets/140389762/5911c62b-91f4-4a3e-9682-8787ddc8ad25)


# 3. 모델 구성

## 3-0. 데이터 준비

< 1. facial expression recognition competition data >

```python
#data set array구조 reshape
shape_x = 48
shape_y = 48

# X_train, y_train, X_test, y_test split
X_train = train_df.iloc[:, 1].values # pixles
y_train = train_df.iloc[:, 0].values # emotion

X_test = test_df.iloc[:, 1].values # pixles
y_test = test_df.iloc[:, 0].values # emotion

# 전체데이터
X = df.iloc[:, 1].values # pixles
y = df.iloc[:, 0].values # emotion

# array([array([....])]) 구조를 바꾸기 위한 np.vstack
X_train = np.vstack(X_train)
X_test = np.vstack(X_test)
X = np.vstack(X)

# 4차원 데이터셋 만들기 (데이터개수, x축, y축, rgb)
X_train_ds = np.reshape(X_train, (X_train.shape[0], shape_x, shape_y, 1))
y_train_ds = np.reshape(y_train, (y_train.shape[0], 1))

X_test_ds = np.reshape(X_test, (X_test.shape[0], shape_x, shape_y, 1))
y_test_ds = np.reshape(y_test, (y_test.shape[0], 1))

print(X_train_ds.shape, y_train_ds.shape)
print(X_test_ds.shape, y_test_ds.shape)

```


## 3-1. 감정분석 모델

º 감정분석 모델의 경우 기본 합성곱 신경망을 사용하였습니다.

º ImageDataGenerator를 사용하여 보다 더 다양한 이미지에 대한 학습이 이루어지도록 하였습니다.

º 기존 이미지 데이터에서 얼굴만을 가져오는 함수를 만들어 성능 개선이 이루어지도록 하였습니다.

< facedetecting 함수 >

```python

def detect_face(frame):
    
    # cascade pre-trained 모델 불러오기
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # RGB를 gray scale로 바꾸기
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # cascade 멀티스케일 분류
    detected_faces = face_cascade.detectMultiScale(gray,
                                                   scaleFactor = 1.1,
                                                   minNeighbors = 6,
                                                   minSize = (shape_x, shape_y),
                                                   flags = cv2.CASCADE_SCALE_IMAGE
                                                  )
    
    coord = []
    for x, y, w, h in detected_faces:
        if w > 100:
            sub_img = frame[y:y+h, x:x+w]
            coord.append([x, y, w, h])
            
    return gray, detected_faces, coord

# 전체 이미지에서 찾아낸 얼굴을 추출하는 함수
def extract_face_features(gray, detected_faces, coord, offset_coefficients=(0.075, 0.05)):
    new_face = []
    for det in detected_faces:
        
        # 얼굴로 감지된 영역
        x, y, w, h = det
        
        # 이미지 경계값 받기
        horizontal_offset = int(np.floor(offset_coefficients[0] * w))
        vertical_offset = int(np.floor(offset_coefficients[1] * h))

        
        # gray scacle 에서 해당 위치 가져오기
        extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]
        
        # 얼굴 이미지만 확대
        new_extracted_face = zoom(extracted_face, (shape_x/extracted_face.shape[0], shape_y/extracted_face.shape[1]))
        new_extracted_face = new_extracted_face.astype(np.float32)
        new_extracted_face /= float(new_extracted_face.max()) # sacled
        new_face.append(new_extracted_face)
        
    return new_face

```

< 감정분석 모델 구성 > 
```python

input = Input(shape=(48, 48, 3))

cnn1 = Conv2D(128, kernel_size=3, activation='relu')(input)
cnn1 = Conv2D(128, kernel_size=3, activation='relu')(cnn1)
cnn1 = Conv2D(128, kernel_size=3, activation='relu')(cnn1)
cnn1 = MaxPool2D(pool_size=3, strides=2)(cnn1)

cnn2 = Conv2D(128, kernel_size=3, activation='relu')(cnn1)
cnn2 = Conv2D(128, kernel_size=3, activation='relu')(cnn2)
cnn2 = Conv2D(128, kernel_size=3, activation='relu')(cnn2)
cnn2 = MaxPool2D(pool_size=3, strides=2)(cnn2)

dense = Flatten()(cnn2)
dense = Dropout(0.2)(dense)
dense = Dense(1024, activation='relu')(dense)
dense = Dense(1024, activation='relu')(dense)

output = Dense(1, activation='linear', name='age')(dense)

model_age = Model(input, output)

# Compile the model
model_age.compile(optimizer=Adam(0.0001), loss='mse', metrics=['mae'])

# Train the model
history = model_age.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

```


## 3-2. 나이판별 모델

< 나이판별 모델 >

```python

# 입력
input_shape = (48, 48, 3)
input_layer = Input(shape=input_shape)

# 모델 구성
input = Input(shape=(48, 48, 3))

cnn1 = Conv2D(128, kernel_size=3, activation='relu')(input)
cnn1 = Conv2D(128, kernel_size=3, activation='relu')(cnn1)
cnn1 = Conv2D(128, kernel_size=3, activation='relu')(cnn1)
cnn1 = MaxPool2D(pool_size=3, strides=2)(cnn1)

cnn2 = Conv2D(128, kernel_size=3, activation='relu')(cnn1)
cnn2 = Conv2D(128, kernel_size=3, activation='relu')(cnn2)
cnn2 = Conv2D(128, kernel_size=3, activation='relu')(cnn2)
cnn2 = MaxPool2D(pool_size=3, strides=2)(cnn2)

dense = Flatten()(cnn2)
dense = Dropout(0.2)(dense)
dense = Dense(1024, activation='relu')(dense)
dense = Dense(1024, activation='relu')(dense)

output = Dense(1, activation='linear', name='age')(dense)

model_age = Model(input, output)

# 모델 병합
model_age.compile(optimizer=Adam(0.0001), loss='mse', metrics=['mae'])

# 모델 학습
history = model_age.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

```

## 3-3. 성별판별 모델

< 성별판별 모델 구성 > 

```python

# 모델 구조
cnn1 = Conv2D(36, kernel_size=3, activation='relu')(input)
cnn1 = MaxPool2D(pool_size=3, strides=2)(cnn1)
cnn2 = Conv2D(64, kernel_size=3, activation='relu')(cnn1)
cnn2 = MaxPool2D(pool_size=3, strides=2)(cnn2)
cnn3 = Conv2D(128, kernel_size=3, activation='relu')(cnn2)
cnn3 = MaxPool2D(pool_size=3, strides=2)(cnn3)

dense = Flatten()(cnn3)
dense = Dropout(0.2)(dense)
dense = Dense(512, activation='relu')(dense)
dense = Dense(512, activation='relu')(dense)
output = Dense(1, activation='sigmoid', name='gender')(dense)
model_sex = Model(input, output)
model_sex.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model_sex.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

```

## 3-4. 웹캠 적용

< 기존 학습한 모델들 불러오기 > 

```python

#model loads 
emotion_model = keras.models.load_model("C:/Users/shj06/DL_data_for/DL/models/model_emotion_prediction.h5")
age_model = keras.models.load_model("C:/Users/shj06/DL_data_for/DL/models/model_age_prediction.h5")
sex_model = keras.models.load_model("C:/Users/shj06/DL_data_for/DL/models/model_sex_prediction.h5")

```

< 웹캠 화면에 바운딩 박스를 추가하고 모델 결과값 표시 >

```python

emotion_dict = {
    0: 'Surprise',
    1: 'Anger',
    2: 'Disgust',
    3: 'Happy',
    4: 'Sadness',
    5: 'Fear',
    6: 'Neutral'
}

# Your face detection function (update it as per your requirement)
def detect_face(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 그레이스케일로 변환
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detect_face(frame)

    for (x, y, w, h) in faces:
        # 원본 컬러 이미지에서 얼굴 부분 추출 (성별 및 나이 예측용)
        color_face = rgb_frame[y:y+h, x:x+w]
        # 원본 그레이스케일 이미지에서 얼굴 부분 추출 (감정 예측용)
        gray_face = gray_frame[y:y+h, x:x+w]

        # 컬러 얼굴 이미지 처리
        resized_color_face = cv2.resize(color_face, (48, 48))
        normalized_color_face = resized_color_face / 255.0
        reshaped_color_face = np.reshape(normalized_color_face, (1, 48, 48, 3))

        # 그레이스케일 얼굴 이미지 처리
        resized_gray_face = cv2.resize(gray_face, (48, 48))
        normalized_gray_face = resized_gray_face / 255.0
        reshaped_gray_face = np.reshape(normalized_gray_face, (1, 48, 48, 1))

        # 예측
        sex_preds = sex_model.predict(reshaped_color_face)
        age_preds = age_model.predict(reshaped_color_face)
        emotion_preds = emotion_model.predict(reshaped_gray_face)

        # 결과 처리
        sex_text = 'Female' if sex_preds[0][0] > 0.5 else 'Male'
        age_text = str(int(age_preds[0][0]))
        emotion_text = emotion_dict[np.argmax(emotion_preds[0])]

        # 얼굴 주변에 사각형 그리기
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # 결과 표시
        cv2.putText(frame, f'Sex: {sex_text}', (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
        cv2.putText(frame, f'Age: {age_text}', (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
        cv2.putText(frame, f'Emotion: {emotion_text}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

```

# 결과물

![결과1](https://github.com/Yoon-Hee-Jae/cnn-deeplearning/assets/140389762/2e2878c0-2b0e-4ea1-becf-1ef2a74f90d7)
![결과2](https://github.com/Yoon-Hee-Jae/cnn-deeplearning/assets/140389762/378ce5e4-5de1-4797-8b74-49f4d9682a27)

웹캠을 통해 사람 얼굴이 인식될 경우 바운딩 박스와 함께 성별, 나이 그리고 감정 분석 결과가 나타나는 것을 볼 수 있음.


