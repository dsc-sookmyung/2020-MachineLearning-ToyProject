# How to install tensorflow-gpu

잡담 : tensorflow-gpu 설치는 두 번째다. 첫 설치 때는 별 문제없이 잘 진행되었다. 이번 두번째 설치를 하면서 그 땐 운이 좋았다고 느꼈다. 그런 의미에서 tensorflow-gpu 설치하는 법을 적는다. 

가장 결정적인 힌트를 준 블로그 : https://coding-groot.tistory.com/87

설치할 때 제일 중요한 것은 **버전**이다.  
버전 확인하기 : https://www.tensorflow.org/install/source_windows?hl=ko#gpu  
버전들을 정하고 모두 현재 다운로드가 가능한지 확인하고 설치를 진행한다. 

***

### 순서
NVIDIA 드라이버 설치 -> CUDA 설치(visual studio 설치) -> cuDNN -> 가상환경 만들기 -> tensorflow-gpu 설치  

### 1. NVIDIA 드라이버 설치
[장치 관리자 > 디스플레이 어댑터]에서 그래픽 드라이버 이름을 확인한다. ex) NVIDIA GeForce GTX 1050  
NVIDIA 드라이버 다운로드 : https://www.nvidia.co.kr/Download/index.aspx?lang=kr  
위 사이트에서 드라이버를 검색해서 다운로드하고 설치한다. 다운로드 타입은 Game Ready 드라이버(GRD)면 된다. 

### 2. CUDA 설치
CUDA 다운로드 : https://developer.nvidia.com/cuda-toolkit-archive  
CUDA는 exe(network) small버전으로 충분하다. 

CUDA 설치 중 visual studio 관련 설명이 나온다면 버전에 맞는 visual studio가 없다는 의미다.  
여기서 버전에 맞는 visual studio 설치하고 [뒤로] > [Next]하면 된다.   

### 3. cuDNN
cuDNN 다운로드 : https://developer.nvidia.com/rdp/cudnn-archive   
cuDNN은 압축 푼 후에 `C:\Users\User\Downloads\cudnn-10.0-windows10-x64-v7.4.2.24\cuda` 하위의 3개 폴더를 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0` 로 drag and drop하면 된다. 

### 4. 가상환경 만들고 tensorflow-gpu 설치
Anaconda에서 가상환경을 만들어 `$conda install tensorflow-gpu==2.0.0` 으로 설치한다.  
설치 후 `$python` `$from tensorflow.python.client import device_lib`  `$device_lib.list_local_devices()` 했을 때 "/device:GPU:0"가 나오면 성공이다. 

***

버전에 맞게 설치하지 않으면  
`from tensorflow.python.client import device_lib`  
`print(device_lib.list_local_devices())`를 실행했을 때 "/device:CPU:0"만 나온다.  
"/device:GPU:0"가 나와야 GPU사용이 가능하다. 

나의 경우 python 3.8에 tensorflow-gpu-2.3.0, MSVC 2019, cuDNN 8.0, CUDA 11을 설치했다.   
"/device:GPU:0"를 output으로 얻기 위해 다양한 시도를 해보다가 버전이 맞지 않다는 사실을 알게 되었다.  
cuDNN 7.4와 CUDA 10.1로 재설치하려했지만, CUDA 10.1에서 작동하는 cuDNN 7.4가 존재하지 않았다.     

python 3.7에 tensorflow-gpu-2.0.0, MSVC 2017, cuDNN 7.4.2, CUDA 10.0로 환경을 갈아엎은 후에야 tensorflow-gpu를 사용할 수 있었다. 
