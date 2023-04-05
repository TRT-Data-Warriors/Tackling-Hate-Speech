# Aşağılayıcı Söylemlerin Doğal Dil İşleme İle Tespiti

## <b>HIZLI BAŞLANGIÇ</b>

### <b>Gereksinimler</b>
Conda paket yöneticisi en son sürümde olmalıdır.

Herhangi bir CUDA/GPU sorunuyla karşılaşmamak için doğru bir biçimde conda ortamları kurulmalı ve requirements.txt içindeki paket versiyonlarına dikkat edilmelidir. Tensorflow GPU için komutlar aşağıdaki gibi uygulanabilir. Tüm kurulum adımları environment.yml içinde verilmiştir. 

```shell
conda create --name=tf_gpu python=3.9
conda activate tf_gpu
conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# restart program/server
conda activate tf_gpu
python3 -m pip install tensorflow==2.10
```

### <b>Ortam Kurulumu</b>
Projeyi sorunsuz çalıştırabilmek için aşağıdaki komutları kullanarak bir virtual environment oluşturun:

```shell
conda env create -f environment.yml
conda activate tdd_acikhack
```
## Veri Artırımı (Data Augmentation)
Training veri setine yaklaşık 5 bin kadar yeni veri eklenmiş ve bu durum performansa 1.5-2 puan pozitif etki yapmıştır.
- Twitter, İnci Sözlük, Ekşi Sözlük sitelerinden toplanan kullanıcı yorumları
- <a href='https://coltekin.github.io/offensive-turkish/'>Offenseval</a> veri seti
- <a href='https://coltekin.github.io/offensive-turkish/'>Çöltekin Troff</a> veri seti 
- ChatGPT
- Google Translate servisi

## <b>Nasıl Çalıştırılır?</b>
```shell
  python run.py
--train_data_path TRAIN_VERISI_ADRESI
--valid_data_path VALIDATION_VERISI_ADRESI   
--max_len 32   
--epochs 20   
--batch_size 256
```
ya da <b>classification.ipynb</b> notebookunu kullanabilirsiniz.

## <b>Tüm Model Deney Sonuçları</b>

| Model | Ortalama F1 Macro Skoru | KFOLD |
| --- | --- | --- |
| 1. TFIDF + Catboost/XGB | ~0.75-0.77 | No
| 2. Fasttext/Word2Vec + BiLSTM/CNN | ~0.87-0.89 | No
| 3. BERTurk (cased, 32k) | 0.9376 | No
| 4. BERTurk (uncased, 32k) | 0.9412 | No 
| 5. ConvBERTurk | 0.9431 | No |
| 6. ConvBERTurk mC4 + Bi-LSTM + Attention| 0.9664 | Yes |
| 7. ConvBERTurk mC4 + Bi-GRU + CNN | 0.9672 | Yes |
| 8. ConvBERTurk mC4 + Bi-LSTM | 0.9674 | Yes |
| 9: Ensemble --> 6, 7 and 8. modeller | 0.97003 | No |


