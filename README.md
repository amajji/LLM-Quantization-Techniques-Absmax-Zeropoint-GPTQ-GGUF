# Absmax, zero-point, GTPQ, GGUF quantization techniques.

Data scientist | [Anass MAJJI](https://www.linkedin.com/in/anass-majji-729773157/)

***


## :monocle_face: Description
LLM are known for their expensive computational requierements. Typically, the memory needed for a model is calculated by multiplying its number of parameters by the precision of these values(data type). However, to reduce memory consumption, numbers can be stored using lower precision data types through a process known as quantization. 
We distinguish two types of weight quatization families : 
- **Post training quantization (PTQ)** : is a technique where weights of an already trained model are converted to lower precision without necessitating any retraining. Although easy to implement, PTQ is associated with potential performance degradation. In this project, we are going to implement, 4 quantization techniques : 
	- Two naive 8-bit quantization techniques : a symmetric one with absolute maximum quantization (absmax) and an asymmetric one with zero-point quantization. In both cases, the goal is to map an FP32 tensor (original weights) to an INT8 tensor (quantized weights).
	- Two noval 4-bit quantization techniques with minimal performance degradation : GPTQ and GGUF : 

		- **GTPQ** : this method uses an asymmetric quantization method and does so layer by layer such that each layer is processed independently before continuing to the next. During this layer-wise quantization process, it first converts the layer's weights into the inverse Hessian. Hessian matrix is a second-order derivative of the model's loss function and it tells us how sensitive the model's output is to changes in each weight. Weights associated with smaller values in the Hessian matrix are more crucial because small changes in these weights can lead to significant changes in the model's perfomrance. This process allows us to calculate the quantization error then redistribute it over the other weights in the row.

		- **GGUF** : Although GPTQ is a great quantization technique to run a full LLM on a GPU, we might not always have that capacity. Instead, we can use GGUF to offload any layer of the LLM to the CPU. This technique consists in splitting a given layer into **super block**, each containing a set of **sub blocks**. We extract the scaling factor from the **super block** and then we quantize the **sub blocks** using absolute maximum quantization technique: the scale factor is calculated using the information from the **sub block** but is quantized using the scale factor of the **super block**.


- **Quantization aware training (QAT)** : this technique consists in converting weights to lower precision during pre-training or finetuning stage resulting in enhenced model performance. 
