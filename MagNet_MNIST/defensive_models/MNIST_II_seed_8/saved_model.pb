Ԭ
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28ޯ
?
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:*
dtype0
?
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:*
dtype0
?
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_5/kernel/m
?
*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_5/bias/m
y
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_6/kernel/m
?
*Adam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_6/bias/m
y
(Adam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_7/kernel/m
?
*Adam/conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_7/bias/m
y
(Adam/conv2d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_5/kernel/v
?
*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_5/bias/v
y
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_6/kernel/v
?
*Adam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_6/bias/v
y
(Adam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_7/kernel/v
?
*Adam/conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_7/bias/v
y
(Adam/conv2d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?%
value?$B?$ B?$
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures

_init_input_shape
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?
iter

beta_1

 beta_2
	!decay
"learning_ratemBmCmDmEmFmGvHvIvJvKvLvM
*
0
1
2
3
4
5
*
0
1
2
3
4
5
 
?
#non_trainable_variables

$layers
%metrics
&layer_regularization_losses
'layer_metrics
	variables
trainable_variables
regularization_losses
 
 
[Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3

70
81
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	9total
	:count
;	variables
<	keras_api
D
	=total
	>count
?
_fn_kwargs
@	variables
A	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

90
:1

;	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

=0
>1

@	variables
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_6/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_6/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_7/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_7/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_6/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_6/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_7/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_7/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_2Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2conv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *-
f(R&
$__inference_signature_wrapper_195949
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/conv2d_5/kernel/m/Read/ReadVariableOp(Adam/conv2d_5/bias/m/Read/ReadVariableOp*Adam/conv2d_6/kernel/m/Read/ReadVariableOp(Adam/conv2d_6/bias/m/Read/ReadVariableOp*Adam/conv2d_7/kernel/m/Read/ReadVariableOp(Adam/conv2d_7/bias/m/Read/ReadVariableOp*Adam/conv2d_5/kernel/v/Read/ReadVariableOp(Adam/conv2d_5/bias/v/Read/ReadVariableOp*Adam/conv2d_6/kernel/v/Read/ReadVariableOp(Adam/conv2d_6/bias/v/Read/ReadVariableOp*Adam/conv2d_7/kernel/v/Read/ReadVariableOp(Adam/conv2d_7/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *(
f#R!
__inference__traced_save_196314
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_5/kernel/mAdam/conv2d_5/bias/mAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/mAdam/conv2d_7/kernel/mAdam/conv2d_7/bias/mAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/vAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/vAdam/conv2d_7/kernel/vAdam/conv2d_7/bias/v*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *+
f&R$
"__inference__traced_restore_196405??
?
?
D__inference_conv2d_5_layer_call_and_return_conditional_losses_195574

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????b
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?Q
?
!__inference__wrapped_model_195517
input_2I
/model_1_conv2d_5_conv2d_readvariableop_resource:>
0model_1_conv2d_5_biasadd_readvariableop_resource:I
/model_1_conv2d_6_conv2d_readvariableop_resource:>
0model_1_conv2d_6_biasadd_readvariableop_resource:I
/model_1_conv2d_7_conv2d_readvariableop_resource:>
0model_1_conv2d_7_biasadd_readvariableop_resource:
identity??'model_1/conv2d_5/BiasAdd/ReadVariableOp?&model_1/conv2d_5/Conv2D/ReadVariableOp?'model_1/conv2d_6/BiasAdd/ReadVariableOp?&model_1/conv2d_6/Conv2D/ReadVariableOp?'model_1/conv2d_7/BiasAdd/ReadVariableOp?&model_1/conv2d_7/Conv2D/ReadVariableOp?
&model_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model_1/conv2d_5/Conv2DConv2Dinput_2.model_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
'model_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_1/conv2d_5/BiasAddBiasAdd model_1/conv2d_5/Conv2D:output:0/model_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
model_1/conv2d_5/SigmoidSigmoid!model_1/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
+model_1/conv2d_5/ActivityRegularizer/SquareSquaremodel_1/conv2d_5/Sigmoid:y:0*
T0*/
_output_shapes
:??????????
*model_1/conv2d_5/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
(model_1/conv2d_5/ActivityRegularizer/SumSum/model_1/conv2d_5/ActivityRegularizer/Square:y:03model_1/conv2d_5/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: o
*model_1/conv2d_5/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
(model_1/conv2d_5/ActivityRegularizer/mulMul3model_1/conv2d_5/ActivityRegularizer/mul/x:output:01model_1/conv2d_5/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: v
*model_1/conv2d_5/ActivityRegularizer/ShapeShapemodel_1/conv2d_5/Sigmoid:y:0*
T0*
_output_shapes
:?
8model_1/conv2d_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:model_1/conv2d_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:model_1/conv2d_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2model_1/conv2d_5/ActivityRegularizer/strided_sliceStridedSlice3model_1/conv2d_5/ActivityRegularizer/Shape:output:0Amodel_1/conv2d_5/ActivityRegularizer/strided_slice/stack:output:0Cmodel_1/conv2d_5/ActivityRegularizer/strided_slice/stack_1:output:0Cmodel_1/conv2d_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
)model_1/conv2d_5/ActivityRegularizer/CastCast;model_1/conv2d_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
,model_1/conv2d_5/ActivityRegularizer/truedivRealDiv,model_1/conv2d_5/ActivityRegularizer/mul:z:0-model_1/conv2d_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
&model_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model_1/conv2d_6/Conv2DConv2Dmodel_1/conv2d_5/Sigmoid:y:0.model_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
'model_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_1/conv2d_6/BiasAddBiasAdd model_1/conv2d_6/Conv2D:output:0/model_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
model_1/conv2d_6/SigmoidSigmoid!model_1/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
+model_1/conv2d_6/ActivityRegularizer/SquareSquaremodel_1/conv2d_6/Sigmoid:y:0*
T0*/
_output_shapes
:??????????
*model_1/conv2d_6/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
(model_1/conv2d_6/ActivityRegularizer/SumSum/model_1/conv2d_6/ActivityRegularizer/Square:y:03model_1/conv2d_6/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: o
*model_1/conv2d_6/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
(model_1/conv2d_6/ActivityRegularizer/mulMul3model_1/conv2d_6/ActivityRegularizer/mul/x:output:01model_1/conv2d_6/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: v
*model_1/conv2d_6/ActivityRegularizer/ShapeShapemodel_1/conv2d_6/Sigmoid:y:0*
T0*
_output_shapes
:?
8model_1/conv2d_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:model_1/conv2d_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:model_1/conv2d_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2model_1/conv2d_6/ActivityRegularizer/strided_sliceStridedSlice3model_1/conv2d_6/ActivityRegularizer/Shape:output:0Amodel_1/conv2d_6/ActivityRegularizer/strided_slice/stack:output:0Cmodel_1/conv2d_6/ActivityRegularizer/strided_slice/stack_1:output:0Cmodel_1/conv2d_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
)model_1/conv2d_6/ActivityRegularizer/CastCast;model_1/conv2d_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
,model_1/conv2d_6/ActivityRegularizer/truedivRealDiv,model_1/conv2d_6/ActivityRegularizer/mul:z:0-model_1/conv2d_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
&model_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model_1/conv2d_7/Conv2DConv2Dmodel_1/conv2d_6/Sigmoid:y:0.model_1/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
'model_1/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_1/conv2d_7/BiasAddBiasAdd model_1/conv2d_7/Conv2D:output:0/model_1/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
model_1/conv2d_7/SigmoidSigmoid!model_1/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
+model_1/conv2d_7/ActivityRegularizer/SquareSquaremodel_1/conv2d_7/Sigmoid:y:0*
T0*/
_output_shapes
:??????????
*model_1/conv2d_7/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
(model_1/conv2d_7/ActivityRegularizer/SumSum/model_1/conv2d_7/ActivityRegularizer/Square:y:03model_1/conv2d_7/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: o
*model_1/conv2d_7/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
(model_1/conv2d_7/ActivityRegularizer/mulMul3model_1/conv2d_7/ActivityRegularizer/mul/x:output:01model_1/conv2d_7/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: v
*model_1/conv2d_7/ActivityRegularizer/ShapeShapemodel_1/conv2d_7/Sigmoid:y:0*
T0*
_output_shapes
:?
8model_1/conv2d_7/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
:model_1/conv2d_7/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
:model_1/conv2d_7/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2model_1/conv2d_7/ActivityRegularizer/strided_sliceStridedSlice3model_1/conv2d_7/ActivityRegularizer/Shape:output:0Amodel_1/conv2d_7/ActivityRegularizer/strided_slice/stack:output:0Cmodel_1/conv2d_7/ActivityRegularizer/strided_slice/stack_1:output:0Cmodel_1/conv2d_7/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
)model_1/conv2d_7/ActivityRegularizer/CastCast;model_1/conv2d_7/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
,model_1/conv2d_7/ActivityRegularizer/truedivRealDiv,model_1/conv2d_7/ActivityRegularizer/mul:z:0-model_1/conv2d_7/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: s
IdentityIdentitymodel_1/conv2d_7/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp(^model_1/conv2d_5/BiasAdd/ReadVariableOp'^model_1/conv2d_5/Conv2D/ReadVariableOp(^model_1/conv2d_6/BiasAdd/ReadVariableOp'^model_1/conv2d_6/Conv2D/ReadVariableOp(^model_1/conv2d_7/BiasAdd/ReadVariableOp'^model_1/conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2R
'model_1/conv2d_5/BiasAdd/ReadVariableOp'model_1/conv2d_5/BiasAdd/ReadVariableOp2P
&model_1/conv2d_5/Conv2D/ReadVariableOp&model_1/conv2d_5/Conv2D/ReadVariableOp2R
'model_1/conv2d_6/BiasAdd/ReadVariableOp'model_1/conv2d_6/BiasAdd/ReadVariableOp2P
&model_1/conv2d_6/Conv2D/ReadVariableOp&model_1/conv2d_6/Conv2D/ReadVariableOp2R
'model_1/conv2d_7/BiasAdd/ReadVariableOp'model_1/conv2d_7/BiasAdd/ReadVariableOp2P
&model_1/conv2d_7/Conv2D/ReadVariableOp&model_1/conv2d_7/Conv2D/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?:
?
C__inference_model_1_layer_call_and_return_conditional_losses_195794

inputs)
conv2d_5_195751:
conv2d_5_195753:)
conv2d_6_195764:
conv2d_6_195766:)
conv2d_7_195777:
conv2d_7_195779:
identity

identity_1

identity_2

identity_3?? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_195751conv2d_5_195753*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_195574?
,conv2d_5/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *9
f4R2
0__inference_conv2d_5_activity_regularizer_195530{
"conv2d_5/ActivityRegularizer/ShapeShape)conv2d_5/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_5/ActivityRegularizer/strided_sliceStridedSlice+conv2d_5/ActivityRegularizer/Shape:output:09conv2d_5/ActivityRegularizer/strided_slice/stack:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_5/ActivityRegularizer/CastCast3conv2d_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_5/ActivityRegularizer/truedivRealDiv5conv2d_5/ActivityRegularizer/PartitionedCall:output:0%conv2d_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_195764conv2d_6_195766*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_195599?
,conv2d_6/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *9
f4R2
0__inference_conv2d_6_activity_regularizer_195543{
"conv2d_6/ActivityRegularizer/ShapeShape)conv2d_6/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_6/ActivityRegularizer/strided_sliceStridedSlice+conv2d_6/ActivityRegularizer/Shape:output:09conv2d_6/ActivityRegularizer/strided_slice/stack:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_6/ActivityRegularizer/CastCast3conv2d_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_6/ActivityRegularizer/truedivRealDiv5conv2d_6/ActivityRegularizer/PartitionedCall:output:0%conv2d_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_195777conv2d_7_195779*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_195624?
,conv2d_7/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *9
f4R2
0__inference_conv2d_7_activity_regularizer_195556{
"conv2d_7/ActivityRegularizer/ShapeShape)conv2d_7/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_7/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_7/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_7/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_7/ActivityRegularizer/strided_sliceStridedSlice+conv2d_7/ActivityRegularizer/Shape:output:09conv2d_7/ActivityRegularizer/strided_slice/stack:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_7/ActivityRegularizer/CastCast3conv2d_7/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_7/ActivityRegularizer/truedivRealDiv5conv2d_7/ActivityRegularizer/PartitionedCall:output:0%conv2d_7/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
IdentityIdentity)conv2d_7/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????h

Identity_1Identity(conv2d_5/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_2Identity(conv2d_6/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_3Identity(conv2d_7/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?L
?
C__inference_model_1_layer_call_and_return_conditional_losses_196053

inputsA
'conv2d_5_conv2d_readvariableop_resource:6
(conv2d_5_biasadd_readvariableop_resource:A
'conv2d_6_conv2d_readvariableop_resource:6
(conv2d_6_biasadd_readvariableop_resource:A
'conv2d_7_conv2d_readvariableop_resource:6
(conv2d_7_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3??conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????p
conv2d_5/SigmoidSigmoidconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????}
#conv2d_5/ActivityRegularizer/SquareSquareconv2d_5/Sigmoid:y:0*
T0*/
_output_shapes
:?????????{
"conv2d_5/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_5/ActivityRegularizer/SumSum'conv2d_5/ActivityRegularizer/Square:y:0+conv2d_5/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_5/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
 conv2d_5/ActivityRegularizer/mulMul+conv2d_5/ActivityRegularizer/mul/x:output:0)conv2d_5/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: f
"conv2d_5/ActivityRegularizer/ShapeShapeconv2d_5/Sigmoid:y:0*
T0*
_output_shapes
:z
0conv2d_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_5/ActivityRegularizer/strided_sliceStridedSlice+conv2d_5/ActivityRegularizer/Shape:output:09conv2d_5/ActivityRegularizer/strided_slice/stack:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_5/ActivityRegularizer/CastCast3conv2d_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_5/ActivityRegularizer/truedivRealDiv$conv2d_5/ActivityRegularizer/mul:z:0%conv2d_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_6/Conv2DConv2Dconv2d_5/Sigmoid:y:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????p
conv2d_6/SigmoidSigmoidconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????}
#conv2d_6/ActivityRegularizer/SquareSquareconv2d_6/Sigmoid:y:0*
T0*/
_output_shapes
:?????????{
"conv2d_6/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_6/ActivityRegularizer/SumSum'conv2d_6/ActivityRegularizer/Square:y:0+conv2d_6/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_6/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
 conv2d_6/ActivityRegularizer/mulMul+conv2d_6/ActivityRegularizer/mul/x:output:0)conv2d_6/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: f
"conv2d_6/ActivityRegularizer/ShapeShapeconv2d_6/Sigmoid:y:0*
T0*
_output_shapes
:z
0conv2d_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_6/ActivityRegularizer/strided_sliceStridedSlice+conv2d_6/ActivityRegularizer/Shape:output:09conv2d_6/ActivityRegularizer/strided_slice/stack:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_6/ActivityRegularizer/CastCast3conv2d_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_6/ActivityRegularizer/truedivRealDiv$conv2d_6/ActivityRegularizer/mul:z:0%conv2d_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_7/Conv2DConv2Dconv2d_6/Sigmoid:y:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????p
conv2d_7/SigmoidSigmoidconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????}
#conv2d_7/ActivityRegularizer/SquareSquareconv2d_7/Sigmoid:y:0*
T0*/
_output_shapes
:?????????{
"conv2d_7/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_7/ActivityRegularizer/SumSum'conv2d_7/ActivityRegularizer/Square:y:0+conv2d_7/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_7/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
 conv2d_7/ActivityRegularizer/mulMul+conv2d_7/ActivityRegularizer/mul/x:output:0)conv2d_7/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: f
"conv2d_7/ActivityRegularizer/ShapeShapeconv2d_7/Sigmoid:y:0*
T0*
_output_shapes
:z
0conv2d_7/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_7/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_7/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_7/ActivityRegularizer/strided_sliceStridedSlice+conv2d_7/ActivityRegularizer/Shape:output:09conv2d_7/ActivityRegularizer/strided_slice/stack:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_7/ActivityRegularizer/CastCast3conv2d_7/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_7/ActivityRegularizer/truedivRealDiv$conv2d_7/ActivityRegularizer/mul:z:0%conv2d_7/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: k
IdentityIdentityconv2d_7/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????h

Identity_1Identity(conv2d_5/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_2Identity(conv2d_6/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_3Identity(conv2d_7/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
(__inference_model_1_layer_call_fn_195832
input_2!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????: : : *(
_read_only_resource_inputs

*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_195794w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?:
?
C__inference_model_1_layer_call_and_return_conditional_losses_195878
input_2)
conv2d_5_195835:
conv2d_5_195837:)
conv2d_6_195848:
conv2d_6_195850:)
conv2d_7_195861:
conv2d_7_195863:
identity

identity_1

identity_2

identity_3?? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_5_195835conv2d_5_195837*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_195574?
,conv2d_5/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *9
f4R2
0__inference_conv2d_5_activity_regularizer_195530{
"conv2d_5/ActivityRegularizer/ShapeShape)conv2d_5/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_5/ActivityRegularizer/strided_sliceStridedSlice+conv2d_5/ActivityRegularizer/Shape:output:09conv2d_5/ActivityRegularizer/strided_slice/stack:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_5/ActivityRegularizer/CastCast3conv2d_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_5/ActivityRegularizer/truedivRealDiv5conv2d_5/ActivityRegularizer/PartitionedCall:output:0%conv2d_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_195848conv2d_6_195850*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_195599?
,conv2d_6/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *9
f4R2
0__inference_conv2d_6_activity_regularizer_195543{
"conv2d_6/ActivityRegularizer/ShapeShape)conv2d_6/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_6/ActivityRegularizer/strided_sliceStridedSlice+conv2d_6/ActivityRegularizer/Shape:output:09conv2d_6/ActivityRegularizer/strided_slice/stack:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_6/ActivityRegularizer/CastCast3conv2d_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_6/ActivityRegularizer/truedivRealDiv5conv2d_6/ActivityRegularizer/PartitionedCall:output:0%conv2d_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_195861conv2d_7_195863*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_195624?
,conv2d_7/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *9
f4R2
0__inference_conv2d_7_activity_regularizer_195556{
"conv2d_7/ActivityRegularizer/ShapeShape)conv2d_7/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_7/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_7/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_7/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_7/ActivityRegularizer/strided_sliceStridedSlice+conv2d_7/ActivityRegularizer/Shape:output:09conv2d_7/ActivityRegularizer/strided_slice/stack:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_7/ActivityRegularizer/CastCast3conv2d_7/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_7/ActivityRegularizer/truedivRealDiv5conv2d_7/ActivityRegularizer/PartitionedCall:output:0%conv2d_7/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
IdentityIdentity)conv2d_7/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????h

Identity_1Identity(conv2d_5/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_2Identity(conv2d_6/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_3Identity(conv2d_7/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
H__inference_conv2d_5_layer_call_and_return_all_conditional_losses_196137

inputs!
unknown:
	unknown_0:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_195574?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *9
f4R2
0__inference_conv2d_5_activity_regularizer_195530w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????X

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
0__inference_conv2d_5_activity_regularizer_195530
x
identity6
SquareSquarex*
T0*
_output_shapes
:9
RankRank
Square:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????G
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
?
?
)__inference_conv2d_6_layer_call_fn_196146

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_195599w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
$__inference_signature_wrapper_195949
input_2!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? **
f%R#
!__inference__wrapped_model_195517w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
D__inference_conv2d_6_layer_call_and_return_conditional_losses_195599

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????b
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_7_layer_call_and_return_conditional_losses_195624

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????b
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
0__inference_conv2d_6_activity_regularizer_195543
x
identity6
SquareSquarex*
T0*
_output_shapes
:9
RankRank
Square:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????G
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
?:
?
C__inference_model_1_layer_call_and_return_conditional_losses_195924
input_2)
conv2d_5_195881:
conv2d_5_195883:)
conv2d_6_195894:
conv2d_6_195896:)
conv2d_7_195907:
conv2d_7_195909:
identity

identity_1

identity_2

identity_3?? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_5_195881conv2d_5_195883*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_195574?
,conv2d_5/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *9
f4R2
0__inference_conv2d_5_activity_regularizer_195530{
"conv2d_5/ActivityRegularizer/ShapeShape)conv2d_5/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_5/ActivityRegularizer/strided_sliceStridedSlice+conv2d_5/ActivityRegularizer/Shape:output:09conv2d_5/ActivityRegularizer/strided_slice/stack:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_5/ActivityRegularizer/CastCast3conv2d_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_5/ActivityRegularizer/truedivRealDiv5conv2d_5/ActivityRegularizer/PartitionedCall:output:0%conv2d_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_195894conv2d_6_195896*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_195599?
,conv2d_6/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *9
f4R2
0__inference_conv2d_6_activity_regularizer_195543{
"conv2d_6/ActivityRegularizer/ShapeShape)conv2d_6/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_6/ActivityRegularizer/strided_sliceStridedSlice+conv2d_6/ActivityRegularizer/Shape:output:09conv2d_6/ActivityRegularizer/strided_slice/stack:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_6/ActivityRegularizer/CastCast3conv2d_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_6/ActivityRegularizer/truedivRealDiv5conv2d_6/ActivityRegularizer/PartitionedCall:output:0%conv2d_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_195907conv2d_7_195909*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_195624?
,conv2d_7/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *9
f4R2
0__inference_conv2d_7_activity_regularizer_195556{
"conv2d_7/ActivityRegularizer/ShapeShape)conv2d_7/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_7/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_7/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_7/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_7/ActivityRegularizer/strided_sliceStridedSlice+conv2d_7/ActivityRegularizer/Shape:output:09conv2d_7/ActivityRegularizer/strided_slice/stack:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_7/ActivityRegularizer/CastCast3conv2d_7/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_7/ActivityRegularizer/truedivRealDiv5conv2d_7/ActivityRegularizer/PartitionedCall:output:0%conv2d_7/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
IdentityIdentity)conv2d_7/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????h

Identity_1Identity(conv2d_5/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_2Identity(conv2d_6/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_3Identity(conv2d_7/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?	
?
(__inference_model_1_layer_call_fn_195969

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????: : : *(
_read_only_resource_inputs

*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_195642w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?:
?
C__inference_model_1_layer_call_and_return_conditional_losses_195642

inputs)
conv2d_5_195575:
conv2d_5_195577:)
conv2d_6_195600:
conv2d_6_195602:)
conv2d_7_195625:
conv2d_7_195627:
identity

identity_1

identity_2

identity_3?? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_195575conv2d_5_195577*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_195574?
,conv2d_5/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *9
f4R2
0__inference_conv2d_5_activity_regularizer_195530{
"conv2d_5/ActivityRegularizer/ShapeShape)conv2d_5/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_5/ActivityRegularizer/strided_sliceStridedSlice+conv2d_5/ActivityRegularizer/Shape:output:09conv2d_5/ActivityRegularizer/strided_slice/stack:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_5/ActivityRegularizer/CastCast3conv2d_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_5/ActivityRegularizer/truedivRealDiv5conv2d_5/ActivityRegularizer/PartitionedCall:output:0%conv2d_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_195600conv2d_6_195602*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_195599?
,conv2d_6/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *9
f4R2
0__inference_conv2d_6_activity_regularizer_195543{
"conv2d_6/ActivityRegularizer/ShapeShape)conv2d_6/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_6/ActivityRegularizer/strided_sliceStridedSlice+conv2d_6/ActivityRegularizer/Shape:output:09conv2d_6/ActivityRegularizer/strided_slice/stack:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_6/ActivityRegularizer/CastCast3conv2d_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_6/ActivityRegularizer/truedivRealDiv5conv2d_6/ActivityRegularizer/PartitionedCall:output:0%conv2d_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_195625conv2d_7_195627*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_195624?
,conv2d_7/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *9
f4R2
0__inference_conv2d_7_activity_regularizer_195556{
"conv2d_7/ActivityRegularizer/ShapeShape)conv2d_7/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_7/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_7/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_7/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_7/ActivityRegularizer/strided_sliceStridedSlice+conv2d_7/ActivityRegularizer/Shape:output:09conv2d_7/ActivityRegularizer/strided_slice/stack:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_7/ActivityRegularizer/CastCast3conv2d_7/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_7/ActivityRegularizer/truedivRealDiv5conv2d_7/ActivityRegularizer/PartitionedCall:output:0%conv2d_7/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
IdentityIdentity)conv2d_7/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????h

Identity_1Identity(conv2d_5/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_2Identity(conv2d_6/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_3Identity(conv2d_7/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_conv2d_7_layer_call_and_return_all_conditional_losses_196177

inputs!
unknown:
	unknown_0:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_195624?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *9
f4R2
0__inference_conv2d_7_activity_regularizer_195556w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????X

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_conv2d_6_layer_call_and_return_all_conditional_losses_196157

inputs!
unknown:
	unknown_0:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_195599?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *9
f4R2
0__inference_conv2d_6_activity_regularizer_195543w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????X

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?<
?
__inference__traced_save_196314
file_prefix.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_conv2d_5_kernel_m_read_readvariableop3
/savev2_adam_conv2d_5_bias_m_read_readvariableop5
1savev2_adam_conv2d_6_kernel_m_read_readvariableop3
/savev2_adam_conv2d_6_bias_m_read_readvariableop5
1savev2_adam_conv2d_7_kernel_m_read_readvariableop3
/savev2_adam_conv2d_7_bias_m_read_readvariableop5
1savev2_adam_conv2d_5_kernel_v_read_readvariableop3
/savev2_adam_conv2d_5_bias_v_read_readvariableop5
1savev2_adam_conv2d_6_kernel_v_read_readvariableop3
/savev2_adam_conv2d_6_bias_v_read_readvariableop5
1savev2_adam_conv2d_7_kernel_v_read_readvariableop3
/savev2_adam_conv2d_7_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_conv2d_5_kernel_m_read_readvariableop/savev2_adam_conv2d_5_bias_m_read_readvariableop1savev2_adam_conv2d_6_kernel_m_read_readvariableop/savev2_adam_conv2d_6_bias_m_read_readvariableop1savev2_adam_conv2d_7_kernel_m_read_readvariableop/savev2_adam_conv2d_7_bias_m_read_readvariableop1savev2_adam_conv2d_5_kernel_v_read_readvariableop/savev2_adam_conv2d_5_bias_v_read_readvariableop1savev2_adam_conv2d_6_kernel_v_read_readvariableop/savev2_adam_conv2d_6_bias_v_read_readvariableop1savev2_adam_conv2d_7_kernel_v_read_readvariableop/savev2_adam_conv2d_7_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::: : : : : : : : : ::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
?
D__inference_conv2d_7_layer_call_and_return_conditional_losses_196210

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????b
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_7_layer_call_fn_196166

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_195624w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?L
?
C__inference_model_1_layer_call_and_return_conditional_losses_196117

inputsA
'conv2d_5_conv2d_readvariableop_resource:6
(conv2d_5_biasadd_readvariableop_resource:A
'conv2d_6_conv2d_readvariableop_resource:6
(conv2d_6_biasadd_readvariableop_resource:A
'conv2d_7_conv2d_readvariableop_resource:6
(conv2d_7_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3??conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????p
conv2d_5/SigmoidSigmoidconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????}
#conv2d_5/ActivityRegularizer/SquareSquareconv2d_5/Sigmoid:y:0*
T0*/
_output_shapes
:?????????{
"conv2d_5/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_5/ActivityRegularizer/SumSum'conv2d_5/ActivityRegularizer/Square:y:0+conv2d_5/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_5/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
 conv2d_5/ActivityRegularizer/mulMul+conv2d_5/ActivityRegularizer/mul/x:output:0)conv2d_5/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: f
"conv2d_5/ActivityRegularizer/ShapeShapeconv2d_5/Sigmoid:y:0*
T0*
_output_shapes
:z
0conv2d_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_5/ActivityRegularizer/strided_sliceStridedSlice+conv2d_5/ActivityRegularizer/Shape:output:09conv2d_5/ActivityRegularizer/strided_slice/stack:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_5/ActivityRegularizer/CastCast3conv2d_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_5/ActivityRegularizer/truedivRealDiv$conv2d_5/ActivityRegularizer/mul:z:0%conv2d_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_6/Conv2DConv2Dconv2d_5/Sigmoid:y:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????p
conv2d_6/SigmoidSigmoidconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????}
#conv2d_6/ActivityRegularizer/SquareSquareconv2d_6/Sigmoid:y:0*
T0*/
_output_shapes
:?????????{
"conv2d_6/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_6/ActivityRegularizer/SumSum'conv2d_6/ActivityRegularizer/Square:y:0+conv2d_6/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_6/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
 conv2d_6/ActivityRegularizer/mulMul+conv2d_6/ActivityRegularizer/mul/x:output:0)conv2d_6/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: f
"conv2d_6/ActivityRegularizer/ShapeShapeconv2d_6/Sigmoid:y:0*
T0*
_output_shapes
:z
0conv2d_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_6/ActivityRegularizer/strided_sliceStridedSlice+conv2d_6/ActivityRegularizer/Shape:output:09conv2d_6/ActivityRegularizer/strided_slice/stack:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_6/ActivityRegularizer/CastCast3conv2d_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_6/ActivityRegularizer/truedivRealDiv$conv2d_6/ActivityRegularizer/mul:z:0%conv2d_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_7/Conv2DConv2Dconv2d_6/Sigmoid:y:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????p
conv2d_7/SigmoidSigmoidconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????}
#conv2d_7/ActivityRegularizer/SquareSquareconv2d_7/Sigmoid:y:0*
T0*/
_output_shapes
:?????????{
"conv2d_7/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_7/ActivityRegularizer/SumSum'conv2d_7/ActivityRegularizer/Square:y:0+conv2d_7/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_7/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
 conv2d_7/ActivityRegularizer/mulMul+conv2d_7/ActivityRegularizer/mul/x:output:0)conv2d_7/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: f
"conv2d_7/ActivityRegularizer/ShapeShapeconv2d_7/Sigmoid:y:0*
T0*
_output_shapes
:z
0conv2d_7/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_7/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_7/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_7/ActivityRegularizer/strided_sliceStridedSlice+conv2d_7/ActivityRegularizer/Shape:output:09conv2d_7/ActivityRegularizer/strided_slice/stack:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_7/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_7/ActivityRegularizer/CastCast3conv2d_7/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_7/ActivityRegularizer/truedivRealDiv$conv2d_7/ActivityRegularizer/mul:z:0%conv2d_7/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: k
IdentityIdentityconv2d_7/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????h

Identity_1Identity(conv2d_5/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_2Identity(conv2d_6/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_3Identity(conv2d_7/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
(__inference_model_1_layer_call_fn_195660
input_2!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????: : : *(
_read_only_resource_inputs

*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_195642w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?	
?
(__inference_model_1_layer_call_fn_195989

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:?????????: : : *(
_read_only_resource_inputs

*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_195794w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?m
?
"__inference__traced_restore_196405
file_prefix:
 assignvariableop_conv2d_5_kernel:.
 assignvariableop_1_conv2d_5_bias:<
"assignvariableop_2_conv2d_6_kernel:.
 assignvariableop_3_conv2d_6_bias:<
"assignvariableop_4_conv2d_7_kernel:.
 assignvariableop_5_conv2d_7_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: D
*assignvariableop_15_adam_conv2d_5_kernel_m:6
(assignvariableop_16_adam_conv2d_5_bias_m:D
*assignvariableop_17_adam_conv2d_6_kernel_m:6
(assignvariableop_18_adam_conv2d_6_bias_m:D
*assignvariableop_19_adam_conv2d_7_kernel_m:6
(assignvariableop_20_adam_conv2d_7_bias_m:D
*assignvariableop_21_adam_conv2d_5_kernel_v:6
(assignvariableop_22_adam_conv2d_5_bias_v:D
*assignvariableop_23_adam_conv2d_6_kernel_v:6
(assignvariableop_24_adam_conv2d_6_bias_v:D
*assignvariableop_25_adam_conv2d_7_kernel_v:6
(assignvariableop_26_adam_conv2d_7_bias_v:
identity_28??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp assignvariableop_conv2d_5_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_5_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_6_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_6_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_7_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_7_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_conv2d_5_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_conv2d_5_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_conv2d_6_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_conv2d_6_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_conv2d_7_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_conv2d_7_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv2d_5_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv2d_5_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv2d_6_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv2d_6_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_7_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_7_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
G
0__inference_conv2d_7_activity_regularizer_195556
x
identity6
SquareSquarex*
T0*
_output_shapes
:9
RankRank
Square:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????G
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
?
?
D__inference_conv2d_6_layer_call_and_return_conditional_losses_196199

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????b
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_5_layer_call_fn_196126

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_195574w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_5_layer_call_and_return_conditional_losses_196188

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????b
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_28
serving_default_input_2:0?????????D
conv2d_78
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?e
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
N__call__
*O&call_and_return_all_conditional_losses
P_default_save_signature"
_tf_keras_network
6
_init_input_shape"
_tf_keras_input_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
?
iter

beta_1

 beta_2
	!decay
"learning_ratemBmCmDmEmFmGvHvIvJvKvLvM"
	optimizer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
#non_trainable_variables

$layers
%metrics
&layer_regularization_losses
'layer_metrics
	variables
trainable_variables
regularization_losses
N__call__
P_default_save_signature
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
,
Wserving_default"
signature_map
 "
trackable_list_wrapper
):'2conv2d_5/kernel
:2conv2d_5/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
Q__call__
Xactivity_regularizer_fn
*R&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_6/kernel
:2conv2d_6/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
S__call__
Zactivity_regularizer_fn
*T&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_7/kernel
:2conv2d_7/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
U__call__
\activity_regularizer_fn
*V&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	9total
	:count
;	variables
<	keras_api"
_tf_keras_metric
^
	=total
	>count
?
_fn_kwargs
@	variables
A	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
90
:1"
trackable_list_wrapper
-
;	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
=0
>1"
trackable_list_wrapper
-
@	variables"
_generic_user_object
.:,2Adam/conv2d_5/kernel/m
 :2Adam/conv2d_5/bias/m
.:,2Adam/conv2d_6/kernel/m
 :2Adam/conv2d_6/bias/m
.:,2Adam/conv2d_7/kernel/m
 :2Adam/conv2d_7/bias/m
.:,2Adam/conv2d_5/kernel/v
 :2Adam/conv2d_5/bias/v
.:,2Adam/conv2d_6/kernel/v
 :2Adam/conv2d_6/bias/v
.:,2Adam/conv2d_7/kernel/v
 :2Adam/conv2d_7/bias/v
?2?
(__inference_model_1_layer_call_fn_195660
(__inference_model_1_layer_call_fn_195969
(__inference_model_1_layer_call_fn_195989
(__inference_model_1_layer_call_fn_195832?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_model_1_layer_call_and_return_conditional_losses_196053
C__inference_model_1_layer_call_and_return_conditional_losses_196117
C__inference_model_1_layer_call_and_return_conditional_losses_195878
C__inference_model_1_layer_call_and_return_conditional_losses_195924?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_195517input_2"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_5_layer_call_fn_196126?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv2d_5_layer_call_and_return_all_conditional_losses_196137?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_6_layer_call_fn_196146?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv2d_6_layer_call_and_return_all_conditional_losses_196157?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_7_layer_call_fn_196166?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv2d_7_layer_call_and_return_all_conditional_losses_196177?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_195949input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_conv2d_5_activity_regularizer_195530?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
D__inference_conv2d_5_layer_call_and_return_conditional_losses_196188?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_conv2d_6_activity_regularizer_195543?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
D__inference_conv2d_6_layer_call_and_return_conditional_losses_196199?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_conv2d_7_activity_regularizer_195556?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
D__inference_conv2d_7_layer_call_and_return_conditional_losses_196210?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_1955178?5
.?+
)?&
input_2?????????
? ";?8
6
conv2d_7*?'
conv2d_7?????????Z
0__inference_conv2d_5_activity_regularizer_195530&?
?
?	
x
? "? ?
H__inference_conv2d_5_layer_call_and_return_all_conditional_losses_196137z7?4
-?*
(?%
inputs?????????
? ";?8
#? 
0?????????
?
?	
1/0 ?
D__inference_conv2d_5_layer_call_and_return_conditional_losses_196188l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
)__inference_conv2d_5_layer_call_fn_196126_7?4
-?*
(?%
inputs?????????
? " ??????????Z
0__inference_conv2d_6_activity_regularizer_195543&?
?
?	
x
? "? ?
H__inference_conv2d_6_layer_call_and_return_all_conditional_losses_196157z7?4
-?*
(?%
inputs?????????
? ";?8
#? 
0?????????
?
?	
1/0 ?
D__inference_conv2d_6_layer_call_and_return_conditional_losses_196199l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
)__inference_conv2d_6_layer_call_fn_196146_7?4
-?*
(?%
inputs?????????
? " ??????????Z
0__inference_conv2d_7_activity_regularizer_195556&?
?
?	
x
? "? ?
H__inference_conv2d_7_layer_call_and_return_all_conditional_losses_196177z7?4
-?*
(?%
inputs?????????
? ";?8
#? 
0?????????
?
?	
1/0 ?
D__inference_conv2d_7_layer_call_and_return_conditional_losses_196210l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
)__inference_conv2d_7_layer_call_fn_196166_7?4
-?*
(?%
inputs?????????
? " ???????????
C__inference_model_1_layer_call_and_return_conditional_losses_195878?@?=
6?3
)?&
input_2?????????
p 

 
? "W?T
#? 
0?????????
-?*
?	
1/0 
?	
1/1 
?	
1/2 ?
C__inference_model_1_layer_call_and_return_conditional_losses_195924?@?=
6?3
)?&
input_2?????????
p

 
? "W?T
#? 
0?????????
-?*
?	
1/0 
?	
1/1 
?	
1/2 ?
C__inference_model_1_layer_call_and_return_conditional_losses_196053???<
5?2
(?%
inputs?????????
p 

 
? "W?T
#? 
0?????????
-?*
?	
1/0 
?	
1/1 
?	
1/2 ?
C__inference_model_1_layer_call_and_return_conditional_losses_196117???<
5?2
(?%
inputs?????????
p

 
? "W?T
#? 
0?????????
-?*
?	
1/0 
?	
1/1 
?	
1/2 ?
(__inference_model_1_layer_call_fn_195660l@?=
6?3
)?&
input_2?????????
p 

 
? " ???????????
(__inference_model_1_layer_call_fn_195832l@?=
6?3
)?&
input_2?????????
p

 
? " ???????????
(__inference_model_1_layer_call_fn_195969k??<
5?2
(?%
inputs?????????
p 

 
? " ???????????
(__inference_model_1_layer_call_fn_195989k??<
5?2
(?%
inputs?????????
p

 
? " ???????????
$__inference_signature_wrapper_195949?C?@
? 
9?6
4
input_2)?&
input_2?????????";?8
6
conv2d_7*?'
conv2d_7?????????