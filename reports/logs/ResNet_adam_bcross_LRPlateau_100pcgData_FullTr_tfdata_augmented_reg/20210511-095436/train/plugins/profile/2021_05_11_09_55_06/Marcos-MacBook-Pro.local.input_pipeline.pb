	J+???@J+???@!J+???@	?F???D??F???D?!?F???D?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$J+???@Zd;?O??A??????@Y???x?&??*	     ??@2?
LIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2??t??Q@!?? ?RvX@)?t??Q@1?? ?RvX@:Preprocessing2?
UIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::Shuffle??p=
ף??!$jx?F???)?p=
ף??1$jx?F???:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch????????!-?=?u~??)????????1-?=?u~??:Preprocessing2F
Iterator::Model{?G?z??!?1?n????){?G?z??1?1?n????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??~j?t??!????B???)??~j?t??1????B???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?F???D?I7@???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Zd;?O??Zd;?O??!Zd;?O??      ??!       "      ??!       *      ??!       2	??????@??????@!??????@:      ??!       B      ??!       J	???x?&?????x?&??!???x?&??R      ??!       Z	???x?&?????x?&??!???x?&??b      ??!       JCPU_ONLYY?F???D?b q7@???X@