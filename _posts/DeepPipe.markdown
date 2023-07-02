---
layout: post
title:  "Deep Pipeline Embeddings for AutoML"
date:   2023-07-02 10:30:16 +0200
categories: jekyll update
---

![alt text](../figures/deeppipe/standard_ml_pipeline.svg "Standard Machine Learning Pipeline.")

{% highlight bash %}
conda create -n deeppipe_env python==3.9
pip install deeppipe_api
{% endhighlight %}

{% highlight python %}
from deeppipe_api.deeppipe import load_data, openml, DeepPipe

task_id = 37
task = openml.tasks.get_task(task_id)
X_train, X_test, y_train, y_test = load_data(task, fold=0)
deep_pipe = DeepPipe(n_iters = 50,  #bo iterations
                    time_limit = 3600 #in seconds
                    )
deep_pipe.fit(X_train, y_train)
y_pred = deep_pipe.predict(X_test)

#Test
score = deep_pipe.score(X_test, y_test)
print("Test acc.:", score)

#print best pipeline
print(deep_pipe.model)
{% endhighlight %}

Check out the [our-paper][our-paper] for more info.

[our-paper]: https://arxiv.org/abs/2305.14009

