	7?A`e??@7?A`e??@!7?A`e??@	=????g?=????g?!=????g?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$7?A`e??@?????K??A?A`???@Y?l??????*	    @J?@2?
LIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2?j?t??B@!\\#]??X@)j?t??B@1\\#]??X@:Preprocessing2?
UIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::Shuffle???"??~??!̈?????)??"??~??1̈?????:Preprocessing2F
Iterator::Model????????!6݈+???)????????16݈+???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???S㥛?!?@???s??)???S㥛?1?@???s??:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch?? ?rh??!~??ԗ<??)?? ?rh??1~??ԗ<??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9=????g?I??&???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????K???????K??!?????K??      ??!       "      ??!       *      ??!       2	?A`???@?A`???@!?A`???@:      ??!       B      ??!       J	?l???????l??????!?l??????R      ??!       Z	?l???????l??????!?l??????b      ??!       JCPU_ONLYY=????g?b q??&???X@