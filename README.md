# shopping-classification-master
카카오 shopping-classification 대회 결과 산출물

실험 환경 : window10, tensorflow 1.4


#######################################################################################
global_constants 3번째 줄에 base_dir을 실행하려는 root directory명과 맞춰야 함
#######################################################################################


1. 학습 할 경우
학습 방법 : 1) logs 디렉터리 삭제 
           2) data 디렉터리 하위에 있는 original 디렉터리에 train,val,test chunk file들을 옮긴후
           3) train.py 실행

학습 과정
  1) split() : h5py 파일에서 product,model,brand,maker,pid,price,label을 가져와서 txt와 numpy로 저장
  2) process_dataset() : 상품명에 대한 전처리 과정
  3) make_vocap() : character에 대한 사전을 만드는 과정(생략가능)
  4) make_train_dataset() : 상품명 + 모델명을 띄어쓰기를 제거하고 캐릭터 단위로 분해해서 training set을 만듦
                            각각 상품명에 맞는 big category, middle category, small category, detail category와 전체 category 정답 셋을 만듦
  5) 위에서 나온 data들을 shuffle 하여 model 학습
  
  
  * split() 과정이 오래 걸리므로, chunk file을 지정해서 여러개 process 돌리는 것을 


2. 예측만 할 경우
#######################################################################################
 1)ckpt 파일 다운로드 link
 => 다운로드 후 logs 디렉터리 밑으로 파일을 옮겨주세요
 2) inference 파일 실행 (44번째 줄과 45번째 line에서 dev, test 선택해서 예측)
#######################################################################################
 
