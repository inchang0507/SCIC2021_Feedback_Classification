# scic2021_Feedback_Classification

1. ����

1-1. ���� �н� ��(RoBERTa)�� �̿��Ͽ� ��ȭ�� ������ �з�

1-2. �����ӿ�ũ
- ��ó�� >> �� �н� (RoBERTa-small, RoBERTa-large[smote ����/������]) >> �ӻ��(�켱������ ���� ����)


2. ��ó�� ����

2-1. ����(PyKoSpacing), ����� ����(py-hanspell)
- ���� �����Ϳ� �����Ͽ� ���ε� �� �н� �����ͷ� ���

2-2. �������ø�(SMOTE)
- �Ҽ� Ŭ����(2~50��) �����Ϳ� ���� SMOTE ���� ���ø� ����� �����Ͽ� Ŭ���� �ұ��� ��ȭ
- �ؽ�Ʈ �����͸� TfidfVectorizer�� ����ȭ ���� SMOTE�� ����
- RoBERTa-large �� �� ���� �� �ϳ����� ������



3. �� �н� �� �ӻ��

3-1. 3���� ���� Ȱ��, ������ 1������ �����͸� �н� (27�� Ŭ����)

3-2. �� �ӻ���Ͽ� ���� ��� ����
- 3���� �� �ټ��� ��ǥ
- ����� ��� �ٸ� ���, ������ �켱������ ���� ����

�� ��ó�� ���� ����� ���� Class, def ���� �ڵ�� .py ���Ϸ� ���� �� import �Ͽ� ���
model ���� �� preprocess ������ ����
Class_Roberta : RoBERTa�� ������ ��Ʈ �ε�
Preprocessor : ���̺� ���ڵ�, ����Ʈ ���� ��ȯ, �׽�Ʈ ������ ��ó��, �������ø�

�� �� �� ��Ű�� ��ó :
github.com/pytorch/fairseq/tree/master/examples/roberta
github.com/haven-jeon/PyKoSpacing
github.com/ssut/py-hanspell

�� ������ �������� ����� �����ʹ� ���� ���ε�, roberta_large �� robert_small ���� �뷮�� ���� �� ���ε�