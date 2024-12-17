# data-versioning

data versioning tool test

**simple test: 해당 레포에 dvc 설정이 되어있으므로 아래로 실행 가능합니다.**


## dvc 설치
```
pip install dvc[all]
```

## dvc 동작


dvc 동작은 기본적으로 git 과 동일합니다.
- dvc 폴더 내에 변화가 생기면 dvc status(git status 처럼)으로 추적
- dvc add 및 dvc push 를 거칩니다.

그 외는 dvc --help 참고바랍니다.


## simple test

setting.py 는 원격 스토리지 (해당 레포에서는 corp-prod의 s3) 설정해주는 파일입니다.
실행 후 자격증명 정보를 입력하면 임시폴더 (aws)내에 파일을 만들고, 해당 파일들을 dvc config설정에 업데이트합니다.
```
python setting.py
```
train 후 스코어와 함께 모델 등록
```
python dvc-train.py
```
등록된 모델의 mlflow run id를 입력하여 추론 동작
```
python dvc-infer.py
```
------------------------------------------------------------------------------------------------------------------------
----

  

# TL;DR

    
만약 새로운 git repo를 만든다면 아래의 과정이 필요합니다.
init 과정은 누군가 dvc를 설정해놓은 repo라면 불필요해보입니다.

### 1. Init 과정  
```
git init
```
git의 기본 설정입니다.

```
dvc init 
```

dvc init 시 .dvc 폴더가 생성됩니다.
.dvc 폴더에는 cache, tmp, config, .gitignore, config.local 등의 파일이 생성됩니다.

이후, 아래 명령어들로 remote 스토리지 설정을 해주면 config 파일에 자동 업데이트 됩니다.

```
dvc remote add -d myremote s3://mybucket/myfolder
dvc remote modify myremote profile myprofile
```

### 2. dvc config 설정 (사전에 s3, minio 등 스토리지 생성 및 권한 설정 필요)

dvc config(.dvc/config)에서 다음과 같이 원격 스토리지 설정을 합니다.

- .dvc/config.local 파일을 생성하여 다음처럼 설정해놓고 .gitignore에 추가한 후에 config에서는 profile이름만으로 관리하기도합니다.
```
['remote "test"']
    configpath = ~/.aws/config
    credentialpath = /home/terry/.aws/credentials
```


지금 각자 설정해놓은 .aws 폴더 내에 profile로 설정할 수도 있으며,
configpath, credentialpath등을 임의로 정하여 설정할 수도 있습니다. 

```
[core]
    remote = test
['remote "test"']
    url = s3://dataversion-test/dvc
    configpath = ./aws/config
    credentialpath = ./aws/credentials
```
or

```
[core]
    remote = test
['remote "test"']
    url = s3://dataversion-test/dvc
    profile = corp-prod
```

### 3. 데이터 변경 시
```
dvc add {file}
git add {file}
dvc push
git commit -m "commit message"
git push
git push 후에는 모델 등록이 필요합니다. (현재 레포에서는 python dvc-train.py)
```

