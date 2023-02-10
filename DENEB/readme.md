## Deneb: Denoiseing after Entropy based Denoising

[Arxiv](https://arxiv.org/abs/2212.01189)  
Sumyeiong Ahn, Se-Young Yun  (KAIST AI)



### Abstract
Improperly constructed datasets can result in inaccurate inferences. For instance, models trained on biased datasets perform poorly in terms of generalization (i.e., dataset bias). Recent debiasing techniques have successfully achieved generalization performance by underestimating easy-to-learn samples (i.e., bias-aligned samples) and highlighting difficult-to-learn samples (i.e., bias-conflicting samples). However, these techniques may fail owing to noisy labels, because the trained model recognizes noisy labels as difficult-to-learn and thus highlights them. In this study, we find that earlier approaches that used the provided labels to quantify difficulty could be affected by the small proportion of noisy labels. Furthermore, we find that running denoising algorithms before debiasing is ineffective because denoising algorithms reduce the impact of difficult-to-learn samples, including valuable bias-conflicting samples. Therefore, we propose an approach called denoising after entropy-based debiasing, i.e., DENEB, which has three main stages. (1) The prejudice model is trained by emphasizing (bias-aligned, clean) samples, which are selected using a Gaussian Mixture Model. (2) Using the per-sample entropy from the output of the prejudice model, the sampling probability of each sample that is proportional to the entropy is computed. (3) The final model is trained using existing denoising algorithms with the mini-batches constructed by following the computed sampling probability. Compared to existing debiasing and denoising algorithms, our method achieves better debiasing performance on multiple benchmarks.


### Run command

Step 1) Hyper-parameter tunning
~~~

python preset_standalone.py --gpu $GPU --bratio 0.01 --nratio 0.1 --dataset colored_mnist --model CONV --opt SGD --alg deneb --search_trial 30 --hyperopt
~~~

Step 2) Run standalone
~~~
python train_mix.py --gpu 0 --bratio 0.01 --nratio 0.1 --dataset colored_mnist --model CONV --opt SGD --alg deneb_gce 
~~~