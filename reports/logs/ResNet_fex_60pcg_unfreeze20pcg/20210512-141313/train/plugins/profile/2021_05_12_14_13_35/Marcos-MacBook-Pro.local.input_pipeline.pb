	?I?C?@?I?C?@!?I?C?@	YҬV??\?YҬV??\?!YҬV??\?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?I?C?@{?G?z??AZd;?C?@Y7?A`????*	    ???@2?
LIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2??C?l??H@!?? ??X@)?C?l??H@1?? ??X@:Preprocessing2?
UIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::Shuffle?m???????!0'k;%???)m???????10'k;%???:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch??? ?r??!?3?<R??)??? ?r??1?3?<R??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???Mb??!=R?E???)???Mb??1=R?E???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?v??/??!k???(??){?G?z??1????_??:Preprocessing2F
Iterator::Model???Q???!%?eS???)?~j?t?x?1t?H?r??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9ZҬV??\?IS?@??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	{?G?z??{?G?z??!{?G?z??      ??!       "      ??!       *      ??!       2	Zd;?C?@Zd;?C?@!Zd;?C?@:      ??!       B      ??!       J	7?A`????7?A`????!7?A`????R      ??!       Z	7?A`????7?A`????!7?A`????b      ??!       JCPU_ONLYYZҬV??\?b qS?@??X@