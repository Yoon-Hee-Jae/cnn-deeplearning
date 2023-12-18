# 웹캠을 통한 감정 분석 프로젝트

# 목차
- 개요
- 사용 데이터
- 모델 구성
  0. 데이터 준비
  1. 감정분석 모델
  2. 나이판별 모델
  3. 성별판별 모델
  4. 웹캠 적용

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

```


## 3-1. 감정분석 모델

º 감정분석 모델의 경우 기본 합성곱 신경망을 사용하였습니다.

º ImageDataGenerator를 사용하여 보다 더 다양한 이미지에 대한 학습이 이루어지도록 하였습니다.

º 기존 이미지 데이터에서 얼굴만을 가져오는 함수를 만들어 성능 개선이 이루어지도록 하였습니다.

< facedetecting 함수 >

```python






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
