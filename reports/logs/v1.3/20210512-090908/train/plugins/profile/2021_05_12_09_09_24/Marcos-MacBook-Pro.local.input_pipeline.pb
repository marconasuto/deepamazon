	???Ըe?@???Ըe?@!???Ըe?@	?????R??????R?!?????R?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???Ըe?@/?$????A?S??{e?@Yj?t???*	    ???@2?
LIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2?{?G?z]@!ś?+?X@){?G?z]@1ś?+?X@:Preprocessing2?
UIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::Shuffle?H?z?G??!?co??8??)H?z?G??1?co??8??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???S㥛?!zY4?`??)???S㥛?1zY4?`??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?&1???!W?7?#??)?~j?t???13R?ǔ?:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch/?$???!?????.??)/?$???1?????.??:Preprocessing2F
Iterator::Model?v??/??!??z????)?~j?t?x?13R??t?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?????R?I7?F??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	/?$????/?$????!/?$????      ??!       "      ??!       *      ??!       2	?S??{e?@?S??{e?@!?S??{e?@:      ??!       B      ??!       J	j?t???j?t???!j?t???R      ??!       Z	j?t???j?t???!j?t???b      ??!       JCPU_ONLYY?????R?b q7?F??X@