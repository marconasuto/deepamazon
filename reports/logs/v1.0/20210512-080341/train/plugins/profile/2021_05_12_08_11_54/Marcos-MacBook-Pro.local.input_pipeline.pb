	P??nR??@P??nR??@!P??nR??@	?W"?Й`??W"?Й`?!?W"?Й`?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$P??nR??@?????K??AT㥛??@Y?Zd;??*	    ?"A2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch??x?&kq@!?>iyF@)??x?&kq@1?>iyF@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?~j?tgq@!z?E?tF@)?~j?tgq@1z?E?tF@:Preprocessing2?
LIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2???~j??J@!?L?!@)??~j??J@1?L?!@:Preprocessing2?
JIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::TensorSlice?#??~j?@!?w????)#??~j?@1?w????:Preprocessing2?
UIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::Shuffle??? ?rh@!????????)?? ?rh@1????????:Preprocessing2F
Iterator::ModelP??niq@!???vF@)j?t???1?k??g??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismV-?gq@!?????tF@)???Q???1?S??u?c?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?W"?Й`?I??^???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????K???????K??!?????K??      ??!       "      ??!       *      ??!       2	T㥛??@T㥛??@!T㥛??@:      ??!       B      ??!       J	?Zd;???Zd;??!?Zd;??R      ??!       Z	?Zd;???Zd;??!?Zd;??b      ??!       JCPU_ONLYY?W"?Й`?b q??^???X@