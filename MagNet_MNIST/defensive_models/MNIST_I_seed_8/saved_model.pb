??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
?
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
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
?
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
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
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28֩
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
dtype0
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:*
dtype0
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
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
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/m
?
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_2/kernel/m
?
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_3/kernel/m
?
*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/m
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_4/kernel/m
?
*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_4/bias/m
y
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/v
?
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_2/kernel/v
?
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_3/kernel/v
?
*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/v
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_4/kernel/v
?
*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_4/bias/v
y
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?;
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?;
value?;B?; B?:
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures

_init_input_shape
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
R
&	variables
'trainable_variables
(regularization_losses
)	keras_api
h

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
h

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
?
6iter

7beta_1

8beta_2
	9decay
:learning_ratemnmompmq mr!ms*mt+mu0mv1mwvxvyvzv{ v|!v}*v~+v0v?1v?
F
0
1
2
3
 4
!5
*6
+7
08
19
F
0
1
2
3
 4
!5
*6
+7
08
19
 
?
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics

	variables
trainable_variables
regularization_losses
 
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
?
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
"	variables
#trainable_variables
$regularization_losses
 
 
 
?
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
&	variables
'trainable_variables
(regularization_losses
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

*0
+1
 
?
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
,	variables
-trainable_variables
.regularization_losses
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11

00
11
 
?
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
2	variables
3trainable_variables
4regularization_losses
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
8
0
1
2
3
4
5
6
7

c0
d1
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
 
 
 
4
	etotal
	fcount
g	variables
h	keras_api
D
	itotal
	jcount
k
_fn_kwargs
l	variables
m	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

e0
f1

g	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

i0
j1

l	variables
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *,
f'R%
#__inference_signature_wrapper_97901
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
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
GPU2*0,1,2,3,4,5,6,7J 8? *'
f"R 
__inference__traced_save_98524
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/mAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/vAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/v*3
Tin,
*2(*
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
GPU2*0,1,2,3,4,5,6,7J 8? **
f%R#
!__inference__traced_restore_98651؆
?
?
G__inference_conv2d_1_layer_call_and_return_all_conditional_losses_98239

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
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_97298?
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
GPU2*0,1,2,3,4,5,6,7J 8? *8
f3R1
/__inference_conv2d_1_activity_regularizer_97191w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????X

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
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?R
?
__inference__traced_save_98524
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableop
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
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*?
value?B?(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::::::: : : : : : : : : ::::::::::::::::::::: 2(
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
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:: 


_output_shapes
::
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 
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
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
:: !

_output_shapes
::,"(
&
_output_shapes
:: #

_output_shapes
::,$(
&
_output_shapes
:: %

_output_shapes
::,&(
&
_output_shapes
:: '

_output_shapes
::(

_output_shapes
: 
?
?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_97382

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
?
?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_97323

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
:?????????*
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
:?????????^
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????b
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_97153
input_1E
+model_conv2d_conv2d_readvariableop_resource::
,model_conv2d_biasadd_readvariableop_resource:G
-model_conv2d_1_conv2d_readvariableop_resource:<
.model_conv2d_1_biasadd_readvariableop_resource:G
-model_conv2d_2_conv2d_readvariableop_resource:<
.model_conv2d_2_biasadd_readvariableop_resource:G
-model_conv2d_3_conv2d_readvariableop_resource:<
.model_conv2d_3_biasadd_readvariableop_resource:G
-model_conv2d_4_conv2d_readvariableop_resource:<
.model_conv2d_4_biasadd_readvariableop_resource:
identity??#model/conv2d/BiasAdd/ReadVariableOp?"model/conv2d/Conv2D/ReadVariableOp?%model/conv2d_1/BiasAdd/ReadVariableOp?$model/conv2d_1/Conv2D/ReadVariableOp?%model/conv2d_2/BiasAdd/ReadVariableOp?$model/conv2d_2/Conv2D/ReadVariableOp?%model/conv2d_3/BiasAdd/ReadVariableOp?$model/conv2d_3/Conv2D/ReadVariableOp?%model/conv2d_4/BiasAdd/ReadVariableOp?$model/conv2d_4/Conv2D/ReadVariableOp?
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model/conv2d/Conv2DConv2Dinput_1*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????x
model/conv2d/SigmoidSigmoidmodel/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
'model/conv2d/ActivityRegularizer/SquareSquaremodel/conv2d/Sigmoid:y:0*
T0*/
_output_shapes
:?????????
&model/conv2d/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
$model/conv2d/ActivityRegularizer/SumSum+model/conv2d/ActivityRegularizer/Square:y:0/model/conv2d/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: k
&model/conv2d/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
$model/conv2d/ActivityRegularizer/mulMul/model/conv2d/ActivityRegularizer/mul/x:output:0-model/conv2d/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: n
&model/conv2d/ActivityRegularizer/ShapeShapemodel/conv2d/Sigmoid:y:0*
T0*
_output_shapes
:~
4model/conv2d/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6model/conv2d/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6model/conv2d/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.model/conv2d/ActivityRegularizer/strided_sliceStridedSlice/model/conv2d/ActivityRegularizer/Shape:output:0=model/conv2d/ActivityRegularizer/strided_slice/stack:output:0?model/conv2d/ActivityRegularizer/strided_slice/stack_1:output:0?model/conv2d/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
%model/conv2d/ActivityRegularizer/CastCast7model/conv2d/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(model/conv2d/ActivityRegularizer/truedivRealDiv(model/conv2d/ActivityRegularizer/mul:z:0)model/conv2d/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
model/average_pooling2d/AvgPoolAvgPoolmodel/conv2d/Sigmoid:y:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
?
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model/conv2d_1/Conv2DConv2D(model/average_pooling2d/AvgPool:output:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????|
model/conv2d_1/SigmoidSigmoidmodel/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
)model/conv2d_1/ActivityRegularizer/SquareSquaremodel/conv2d_1/Sigmoid:y:0*
T0*/
_output_shapes
:??????????
(model/conv2d_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
&model/conv2d_1/ActivityRegularizer/SumSum-model/conv2d_1/ActivityRegularizer/Square:y:01model/conv2d_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: m
(model/conv2d_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
&model/conv2d_1/ActivityRegularizer/mulMul1model/conv2d_1/ActivityRegularizer/mul/x:output:0/model/conv2d_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: r
(model/conv2d_1/ActivityRegularizer/ShapeShapemodel/conv2d_1/Sigmoid:y:0*
T0*
_output_shapes
:?
6model/conv2d_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8model/conv2d_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8model/conv2d_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0model/conv2d_1/ActivityRegularizer/strided_sliceStridedSlice1model/conv2d_1/ActivityRegularizer/Shape:output:0?model/conv2d_1/ActivityRegularizer/strided_slice/stack:output:0Amodel/conv2d_1/ActivityRegularizer/strided_slice/stack_1:output:0Amodel/conv2d_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
'model/conv2d_1/ActivityRegularizer/CastCast9model/conv2d_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
*model/conv2d_1/ActivityRegularizer/truedivRealDiv*model/conv2d_1/ActivityRegularizer/mul:z:0+model/conv2d_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model/conv2d_2/Conv2DConv2Dmodel/conv2d_1/Sigmoid:y:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????|
model/conv2d_2/SigmoidSigmoidmodel/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
)model/conv2d_2/ActivityRegularizer/SquareSquaremodel/conv2d_2/Sigmoid:y:0*
T0*/
_output_shapes
:??????????
(model/conv2d_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
&model/conv2d_2/ActivityRegularizer/SumSum-model/conv2d_2/ActivityRegularizer/Square:y:01model/conv2d_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: m
(model/conv2d_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
&model/conv2d_2/ActivityRegularizer/mulMul1model/conv2d_2/ActivityRegularizer/mul/x:output:0/model/conv2d_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: r
(model/conv2d_2/ActivityRegularizer/ShapeShapemodel/conv2d_2/Sigmoid:y:0*
T0*
_output_shapes
:?
6model/conv2d_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8model/conv2d_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8model/conv2d_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0model/conv2d_2/ActivityRegularizer/strided_sliceStridedSlice1model/conv2d_2/ActivityRegularizer/Shape:output:0?model/conv2d_2/ActivityRegularizer/strided_slice/stack:output:0Amodel/conv2d_2/ActivityRegularizer/strided_slice/stack_1:output:0Amodel/conv2d_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
'model/conv2d_2/ActivityRegularizer/CastCast9model/conv2d_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
*model/conv2d_2/ActivityRegularizer/truedivRealDiv*model/conv2d_2/ActivityRegularizer/mul:z:0+model/conv2d_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: j
model/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      l
model/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ?
model/up_sampling2d/mulMul"model/up_sampling2d/Const:output:0$model/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:?
0model/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbormodel/conv2d_2/Sigmoid:y:0model/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(?
$model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model/conv2d_3/Conv2DConv2DAmodel/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0,model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
%model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/conv2d_3/BiasAddBiasAddmodel/conv2d_3/Conv2D:output:0-model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????|
model/conv2d_3/SigmoidSigmoidmodel/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
)model/conv2d_3/ActivityRegularizer/SquareSquaremodel/conv2d_3/Sigmoid:y:0*
T0*/
_output_shapes
:??????????
(model/conv2d_3/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
&model/conv2d_3/ActivityRegularizer/SumSum-model/conv2d_3/ActivityRegularizer/Square:y:01model/conv2d_3/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: m
(model/conv2d_3/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
&model/conv2d_3/ActivityRegularizer/mulMul1model/conv2d_3/ActivityRegularizer/mul/x:output:0/model/conv2d_3/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: r
(model/conv2d_3/ActivityRegularizer/ShapeShapemodel/conv2d_3/Sigmoid:y:0*
T0*
_output_shapes
:?
6model/conv2d_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8model/conv2d_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8model/conv2d_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0model/conv2d_3/ActivityRegularizer/strided_sliceStridedSlice1model/conv2d_3/ActivityRegularizer/Shape:output:0?model/conv2d_3/ActivityRegularizer/strided_slice/stack:output:0Amodel/conv2d_3/ActivityRegularizer/strided_slice/stack_1:output:0Amodel/conv2d_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
'model/conv2d_3/ActivityRegularizer/CastCast9model/conv2d_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
*model/conv2d_3/ActivityRegularizer/truedivRealDiv*model/conv2d_3/ActivityRegularizer/mul:z:0+model/conv2d_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
$model/conv2d_4/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model/conv2d_4/Conv2DConv2Dmodel/conv2d_3/Sigmoid:y:0,model/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
%model/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/conv2d_4/BiasAddBiasAddmodel/conv2d_4/Conv2D:output:0-model/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????|
model/conv2d_4/SigmoidSigmoidmodel/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
)model/conv2d_4/ActivityRegularizer/SquareSquaremodel/conv2d_4/Sigmoid:y:0*
T0*/
_output_shapes
:??????????
(model/conv2d_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
&model/conv2d_4/ActivityRegularizer/SumSum-model/conv2d_4/ActivityRegularizer/Square:y:01model/conv2d_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: m
(model/conv2d_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
&model/conv2d_4/ActivityRegularizer/mulMul1model/conv2d_4/ActivityRegularizer/mul/x:output:0/model/conv2d_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: r
(model/conv2d_4/ActivityRegularizer/ShapeShapemodel/conv2d_4/Sigmoid:y:0*
T0*
_output_shapes
:?
6model/conv2d_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8model/conv2d_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8model/conv2d_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0model/conv2d_4/ActivityRegularizer/strided_sliceStridedSlice1model/conv2d_4/ActivityRegularizer/Shape:output:0?model/conv2d_4/ActivityRegularizer/strided_slice/stack:output:0Amodel/conv2d_4/ActivityRegularizer/strided_slice/stack_1:output:0Amodel/conv2d_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
'model/conv2d_4/ActivityRegularizer/CastCast9model/conv2d_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
*model/conv2d_4/ActivityRegularizer/truedivRealDiv*model/conv2d_4/ActivityRegularizer/mul:z:0+model/conv2d_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: q
IdentityIdentitymodel/conv2d_4/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp&^model/conv2d_3/BiasAdd/ReadVariableOp%^model/conv2d_3/Conv2D/ReadVariableOp&^model/conv2d_4/BiasAdd/ReadVariableOp%^model/conv2d_4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2N
%model/conv2d_3/BiasAdd/ReadVariableOp%model/conv2d_3/BiasAdd/ReadVariableOp2L
$model/conv2d_3/Conv2D/ReadVariableOp$model/conv2d_3/Conv2D/ReadVariableOp2N
%model/conv2d_4/BiasAdd/ReadVariableOp%model/conv2d_4/BiasAdd/ReadVariableOp2L
$model/conv2d_4/Conv2D/ReadVariableOp$model/conv2d_4/Conv2D/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
%__inference_model_layer_call_fn_97430
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout

2*
_collective_manager_ids
 *9
_output_shapes'
%:?????????: : : : : *,
_read_only_resource_inputs

	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_97402w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
&__inference_conv2d_layer_call_fn_98188

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
GPU2*0,1,2,3,4,5,6,7J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_97267w
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
?
h
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_97285

inputs
identity?
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
`
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
I
-__inference_up_sampling2d_layer_call_fn_98269

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *Q
fLRJ
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_97344h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_97344

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"      X
Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Q
mulMulConst:output:0Const_1:output:0*
T0*
_output_shapes
:?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(}
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
D
-__inference_conv2d_activity_regularizer_97166
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

?
%__inference_model_layer_call_fn_97931

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout

2*
_collective_manager_ids
 *9
_output_shapes'
%:?????????: : : : : *,
_read_only_resource_inputs

	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_97402w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
A__inference_conv2d_layer_call_and_return_conditional_losses_97267

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
?d
?
@__inference_model_layer_call_and_return_conditional_losses_97658

inputs&
conv2d_97585:
conv2d_97587:(
conv2d_1_97599:
conv2d_1_97601:(
conv2d_2_97612:
conv2d_2_97614:(
conv2d_3_97626:
conv2d_3_97628:(
conv2d_4_97639:
conv2d_4_97641:
identity

identity_1

identity_2

identity_3

identity_4

identity_5??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_97585conv2d_97587*
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
GPU2*0,1,2,3,4,5,6,7J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_97267?
*conv2d/ActivityRegularizer/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU2*0,1,2,3,4,5,6,7J 8? *6
f1R/
-__inference_conv2d_activity_regularizer_97166w
 conv2d/ActivityRegularizer/ShapeShape'conv2d/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:x
.conv2d/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv2d/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv2d/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(conv2d/ActivityRegularizer/strided_sliceStridedSlice)conv2d/ActivityRegularizer/Shape:output:07conv2d/ActivityRegularizer/strided_slice/stack:output:09conv2d/ActivityRegularizer/strided_slice/stack_1:output:09conv2d/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d/ActivityRegularizer/CastCast1conv2d/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
"conv2d/ActivityRegularizer/truedivRealDiv3conv2d/ActivityRegularizer/PartitionedCall:output:0#conv2d/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
!average_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_97285?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0conv2d_1_97599conv2d_1_97601*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_97298?
,conv2d_1/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU2*0,1,2,3,4,5,6,7J 8? *8
f3R1
/__inference_conv2d_1_activity_regularizer_97191{
"conv2d_1/ActivityRegularizer/ShapeShape)conv2d_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_1/ActivityRegularizer/strided_sliceStridedSlice+conv2d_1/ActivityRegularizer/Shape:output:09conv2d_1/ActivityRegularizer/strided_slice/stack:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_1/ActivityRegularizer/CastCast3conv2d_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_1/ActivityRegularizer/truedivRealDiv5conv2d_1/ActivityRegularizer/PartitionedCall:output:0%conv2d_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_97612conv2d_2_97614*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_97323?
,conv2d_2/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
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
GPU2*0,1,2,3,4,5,6,7J 8? *8
f3R1
/__inference_conv2d_2_activity_regularizer_97204{
"conv2d_2/ActivityRegularizer/ShapeShape)conv2d_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_2/ActivityRegularizer/strided_sliceStridedSlice+conv2d_2/ActivityRegularizer/Shape:output:09conv2d_2/ActivityRegularizer/strided_slice/stack:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_2/ActivityRegularizer/CastCast3conv2d_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_2/ActivityRegularizer/truedivRealDiv5conv2d_2/ActivityRegularizer/PartitionedCall:output:0%conv2d_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
up_sampling2d/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *Q
fLRJ
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_97344?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_3_97626conv2d_3_97628*
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
GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_97357?
,conv2d_3/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
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
GPU2*0,1,2,3,4,5,6,7J 8? *8
f3R1
/__inference_conv2d_3_activity_regularizer_97236{
"conv2d_3/ActivityRegularizer/ShapeShape)conv2d_3/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_3/ActivityRegularizer/strided_sliceStridedSlice+conv2d_3/ActivityRegularizer/Shape:output:09conv2d_3/ActivityRegularizer/strided_slice/stack:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_3/ActivityRegularizer/CastCast3conv2d_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_3/ActivityRegularizer/truedivRealDiv5conv2d_3/ActivityRegularizer/PartitionedCall:output:0%conv2d_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_97639conv2d_4_97641*
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
GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_97382?
,conv2d_4/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
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
GPU2*0,1,2,3,4,5,6,7J 8? *8
f3R1
/__inference_conv2d_4_activity_regularizer_97249{
"conv2d_4/ActivityRegularizer/ShapeShape)conv2d_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_4/ActivityRegularizer/strided_sliceStridedSlice+conv2d_4/ActivityRegularizer/Shape:output:09conv2d_4/ActivityRegularizer/strided_slice/stack:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_4/ActivityRegularizer/CastCast3conv2d_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_4/ActivityRegularizer/truedivRealDiv5conv2d_4/ActivityRegularizer/PartitionedCall:output:0%conv2d_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????f

Identity_1Identity&conv2d/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_2Identity(conv2d_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_3Identity(conv2d_2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_4Identity(conv2d_3/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_5Identity(conv2d_4/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
/__inference_conv2d_2_activity_regularizer_97204
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

?
%__inference_model_layer_call_fn_97961

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout

2*
_collective_manager_ids
 *9
_output_shapes'
%:?????????: : : : : *,
_read_only_resource_inputs

	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_97658w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_conv2d_1_layer_call_fn_98228

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
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_97298w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
/__inference_conv2d_1_activity_regularizer_97191
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
??
?
@__inference_model_layer_call_and_return_conditional_losses_98070

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:A
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:A
'conv2d_3_conv2d_readvariableop_resource:6
(conv2d_3_biasadd_readvariableop_resource:A
'conv2d_4_conv2d_readvariableop_resource:6
(conv2d_4_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????l
conv2d/SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????y
!conv2d/ActivityRegularizer/SquareSquareconv2d/Sigmoid:y:0*
T0*/
_output_shapes
:?????????y
 conv2d/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2d/ActivityRegularizer/SumSum%conv2d/ActivityRegularizer/Square:y:0)conv2d/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: e
 conv2d/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
conv2d/ActivityRegularizer/mulMul)conv2d/ActivityRegularizer/mul/x:output:0'conv2d/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: b
 conv2d/ActivityRegularizer/ShapeShapeconv2d/Sigmoid:y:0*
T0*
_output_shapes
:x
.conv2d/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv2d/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv2d/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(conv2d/ActivityRegularizer/strided_sliceStridedSlice)conv2d/ActivityRegularizer/Shape:output:07conv2d/ActivityRegularizer/strided_slice/stack:output:09conv2d/ActivityRegularizer/strided_slice/stack_1:output:09conv2d/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d/ActivityRegularizer/CastCast1conv2d/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
"conv2d/ActivityRegularizer/truedivRealDiv"conv2d/ActivityRegularizer/mul:z:0#conv2d/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
average_pooling2d/AvgPoolAvgPoolconv2d/Sigmoid:y:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2D"average_pooling2d/AvgPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????p
conv2d_1/SigmoidSigmoidconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????}
#conv2d_1/ActivityRegularizer/SquareSquareconv2d_1/Sigmoid:y:0*
T0*/
_output_shapes
:?????????{
"conv2d_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_1/ActivityRegularizer/SumSum'conv2d_1/ActivityRegularizer/Square:y:0+conv2d_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
 conv2d_1/ActivityRegularizer/mulMul+conv2d_1/ActivityRegularizer/mul/x:output:0)conv2d_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: f
"conv2d_1/ActivityRegularizer/ShapeShapeconv2d_1/Sigmoid:y:0*
T0*
_output_shapes
:z
0conv2d_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_1/ActivityRegularizer/strided_sliceStridedSlice+conv2d_1/ActivityRegularizer/Shape:output:09conv2d_1/ActivityRegularizer/strided_slice/stack:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_1/ActivityRegularizer/CastCast3conv2d_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_1/ActivityRegularizer/truedivRealDiv$conv2d_1/ActivityRegularizer/mul:z:0%conv2d_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_2/Conv2DConv2Dconv2d_1/Sigmoid:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????p
conv2d_2/SigmoidSigmoidconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????}
#conv2d_2/ActivityRegularizer/SquareSquareconv2d_2/Sigmoid:y:0*
T0*/
_output_shapes
:?????????{
"conv2d_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_2/ActivityRegularizer/SumSum'conv2d_2/ActivityRegularizer/Square:y:0+conv2d_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
 conv2d_2/ActivityRegularizer/mulMul+conv2d_2/ActivityRegularizer/mul/x:output:0)conv2d_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: f
"conv2d_2/ActivityRegularizer/ShapeShapeconv2d_2/Sigmoid:y:0*
T0*
_output_shapes
:z
0conv2d_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_2/ActivityRegularizer/strided_sliceStridedSlice+conv2d_2/ActivityRegularizer/Shape:output:09conv2d_2/ActivityRegularizer/strided_slice/stack:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_2/ActivityRegularizer/CastCast3conv2d_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_2/ActivityRegularizer/truedivRealDiv$conv2d_2/ActivityRegularizer/mul:z:0%conv2d_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: d
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      f
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      {
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_2/Sigmoid:y:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_3/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????p
conv2d_3/SigmoidSigmoidconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????}
#conv2d_3/ActivityRegularizer/SquareSquareconv2d_3/Sigmoid:y:0*
T0*/
_output_shapes
:?????????{
"conv2d_3/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_3/ActivityRegularizer/SumSum'conv2d_3/ActivityRegularizer/Square:y:0+conv2d_3/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_3/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
 conv2d_3/ActivityRegularizer/mulMul+conv2d_3/ActivityRegularizer/mul/x:output:0)conv2d_3/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: f
"conv2d_3/ActivityRegularizer/ShapeShapeconv2d_3/Sigmoid:y:0*
T0*
_output_shapes
:z
0conv2d_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_3/ActivityRegularizer/strided_sliceStridedSlice+conv2d_3/ActivityRegularizer/Shape:output:09conv2d_3/ActivityRegularizer/strided_slice/stack:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_3/ActivityRegularizer/CastCast3conv2d_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_3/ActivityRegularizer/truedivRealDiv$conv2d_3/ActivityRegularizer/mul:z:0%conv2d_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_4/Conv2DConv2Dconv2d_3/Sigmoid:y:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????p
conv2d_4/SigmoidSigmoidconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????}
#conv2d_4/ActivityRegularizer/SquareSquareconv2d_4/Sigmoid:y:0*
T0*/
_output_shapes
:?????????{
"conv2d_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_4/ActivityRegularizer/SumSum'conv2d_4/ActivityRegularizer/Square:y:0+conv2d_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
 conv2d_4/ActivityRegularizer/mulMul+conv2d_4/ActivityRegularizer/mul/x:output:0)conv2d_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: f
"conv2d_4/ActivityRegularizer/ShapeShapeconv2d_4/Sigmoid:y:0*
T0*
_output_shapes
:z
0conv2d_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_4/ActivityRegularizer/strided_sliceStridedSlice+conv2d_4/ActivityRegularizer/Shape:output:09conv2d_4/ActivityRegularizer/strided_slice/stack:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_4/ActivityRegularizer/CastCast3conv2d_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_4/ActivityRegularizer/truedivRealDiv$conv2d_4/ActivityRegularizer/mul:z:0%conv2d_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: k
IdentityIdentityconv2d_4/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????f

Identity_1Identity&conv2d/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_2Identity(conv2d_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_3Identity(conv2d_2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_4Identity(conv2d_3/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_5Identity(conv2d_4/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_97357

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
F
/__inference_conv2d_4_activity_regularizer_97249
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
?
h
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_97175

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_layer_call_and_return_all_conditional_losses_98199

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
GPU2*0,1,2,3,4,5,6,7J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_97267?
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
GPU2*0,1,2,3,4,5,6,7J 8? *6
f1R/
-__inference_conv2d_activity_regularizer_97166w
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
?
d
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_98281

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_98179

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:A
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:A
'conv2d_3_conv2d_readvariableop_resource:6
(conv2d_3_biasadd_readvariableop_resource:A
'conv2d_4_conv2d_readvariableop_resource:6
(conv2d_4_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????l
conv2d/SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????y
!conv2d/ActivityRegularizer/SquareSquareconv2d/Sigmoid:y:0*
T0*/
_output_shapes
:?????????y
 conv2d/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2d/ActivityRegularizer/SumSum%conv2d/ActivityRegularizer/Square:y:0)conv2d/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: e
 conv2d/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
conv2d/ActivityRegularizer/mulMul)conv2d/ActivityRegularizer/mul/x:output:0'conv2d/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: b
 conv2d/ActivityRegularizer/ShapeShapeconv2d/Sigmoid:y:0*
T0*
_output_shapes
:x
.conv2d/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv2d/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv2d/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(conv2d/ActivityRegularizer/strided_sliceStridedSlice)conv2d/ActivityRegularizer/Shape:output:07conv2d/ActivityRegularizer/strided_slice/stack:output:09conv2d/ActivityRegularizer/strided_slice/stack_1:output:09conv2d/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d/ActivityRegularizer/CastCast1conv2d/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
"conv2d/ActivityRegularizer/truedivRealDiv"conv2d/ActivityRegularizer/mul:z:0#conv2d/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
average_pooling2d/AvgPoolAvgPoolconv2d/Sigmoid:y:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2D"average_pooling2d/AvgPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????p
conv2d_1/SigmoidSigmoidconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????}
#conv2d_1/ActivityRegularizer/SquareSquareconv2d_1/Sigmoid:y:0*
T0*/
_output_shapes
:?????????{
"conv2d_1/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_1/ActivityRegularizer/SumSum'conv2d_1/ActivityRegularizer/Square:y:0+conv2d_1/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
 conv2d_1/ActivityRegularizer/mulMul+conv2d_1/ActivityRegularizer/mul/x:output:0)conv2d_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: f
"conv2d_1/ActivityRegularizer/ShapeShapeconv2d_1/Sigmoid:y:0*
T0*
_output_shapes
:z
0conv2d_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_1/ActivityRegularizer/strided_sliceStridedSlice+conv2d_1/ActivityRegularizer/Shape:output:09conv2d_1/ActivityRegularizer/strided_slice/stack:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_1/ActivityRegularizer/CastCast3conv2d_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_1/ActivityRegularizer/truedivRealDiv$conv2d_1/ActivityRegularizer/mul:z:0%conv2d_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_2/Conv2DConv2Dconv2d_1/Sigmoid:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????p
conv2d_2/SigmoidSigmoidconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????}
#conv2d_2/ActivityRegularizer/SquareSquareconv2d_2/Sigmoid:y:0*
T0*/
_output_shapes
:?????????{
"conv2d_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_2/ActivityRegularizer/SumSum'conv2d_2/ActivityRegularizer/Square:y:0+conv2d_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
 conv2d_2/ActivityRegularizer/mulMul+conv2d_2/ActivityRegularizer/mul/x:output:0)conv2d_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: f
"conv2d_2/ActivityRegularizer/ShapeShapeconv2d_2/Sigmoid:y:0*
T0*
_output_shapes
:z
0conv2d_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_2/ActivityRegularizer/strided_sliceStridedSlice+conv2d_2/ActivityRegularizer/Shape:output:09conv2d_2/ActivityRegularizer/strided_slice/stack:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_2/ActivityRegularizer/CastCast3conv2d_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_2/ActivityRegularizer/truedivRealDiv$conv2d_2/ActivityRegularizer/mul:z:0%conv2d_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: d
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      f
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      {
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_2/Sigmoid:y:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_3/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????p
conv2d_3/SigmoidSigmoidconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????}
#conv2d_3/ActivityRegularizer/SquareSquareconv2d_3/Sigmoid:y:0*
T0*/
_output_shapes
:?????????{
"conv2d_3/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_3/ActivityRegularizer/SumSum'conv2d_3/ActivityRegularizer/Square:y:0+conv2d_3/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_3/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
 conv2d_3/ActivityRegularizer/mulMul+conv2d_3/ActivityRegularizer/mul/x:output:0)conv2d_3/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: f
"conv2d_3/ActivityRegularizer/ShapeShapeconv2d_3/Sigmoid:y:0*
T0*
_output_shapes
:z
0conv2d_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_3/ActivityRegularizer/strided_sliceStridedSlice+conv2d_3/ActivityRegularizer/Shape:output:09conv2d_3/ActivityRegularizer/strided_slice/stack:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_3/ActivityRegularizer/CastCast3conv2d_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_3/ActivityRegularizer/truedivRealDiv$conv2d_3/ActivityRegularizer/mul:z:0%conv2d_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_4/Conv2DConv2Dconv2d_3/Sigmoid:y:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????p
conv2d_4/SigmoidSigmoidconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????}
#conv2d_4/ActivityRegularizer/SquareSquareconv2d_4/Sigmoid:y:0*
T0*/
_output_shapes
:?????????{
"conv2d_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_4/ActivityRegularizer/SumSum'conv2d_4/ActivityRegularizer/Square:y:0+conv2d_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: g
"conv2d_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *_p?0?
 conv2d_4/ActivityRegularizer/mulMul+conv2d_4/ActivityRegularizer/mul/x:output:0)conv2d_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: f
"conv2d_4/ActivityRegularizer/ShapeShapeconv2d_4/Sigmoid:y:0*
T0*
_output_shapes
:z
0conv2d_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_4/ActivityRegularizer/strided_sliceStridedSlice+conv2d_4/ActivityRegularizer/Shape:output:09conv2d_4/ActivityRegularizer/strided_slice/stack:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_4/ActivityRegularizer/CastCast3conv2d_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_4/ActivityRegularizer/truedivRealDiv$conv2d_4/ActivityRegularizer/mul:z:0%conv2d_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: k
IdentityIdentityconv2d_4/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????f

Identity_1Identity&conv2d/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_2Identity(conv2d_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_3Identity(conv2d_2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_4Identity(conv2d_3/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_5Identity(conv2d_4/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?d
?
@__inference_model_layer_call_and_return_conditional_losses_97402

inputs&
conv2d_97268:
conv2d_97270:(
conv2d_1_97299:
conv2d_1_97301:(
conv2d_2_97324:
conv2d_2_97326:(
conv2d_3_97358:
conv2d_3_97360:(
conv2d_4_97383:
conv2d_4_97385:
identity

identity_1

identity_2

identity_3

identity_4

identity_5??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_97268conv2d_97270*
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
GPU2*0,1,2,3,4,5,6,7J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_97267?
*conv2d/ActivityRegularizer/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU2*0,1,2,3,4,5,6,7J 8? *6
f1R/
-__inference_conv2d_activity_regularizer_97166w
 conv2d/ActivityRegularizer/ShapeShape'conv2d/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:x
.conv2d/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv2d/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv2d/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(conv2d/ActivityRegularizer/strided_sliceStridedSlice)conv2d/ActivityRegularizer/Shape:output:07conv2d/ActivityRegularizer/strided_slice/stack:output:09conv2d/ActivityRegularizer/strided_slice/stack_1:output:09conv2d/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d/ActivityRegularizer/CastCast1conv2d/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
"conv2d/ActivityRegularizer/truedivRealDiv3conv2d/ActivityRegularizer/PartitionedCall:output:0#conv2d/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
!average_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_97285?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0conv2d_1_97299conv2d_1_97301*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_97298?
,conv2d_1/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU2*0,1,2,3,4,5,6,7J 8? *8
f3R1
/__inference_conv2d_1_activity_regularizer_97191{
"conv2d_1/ActivityRegularizer/ShapeShape)conv2d_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_1/ActivityRegularizer/strided_sliceStridedSlice+conv2d_1/ActivityRegularizer/Shape:output:09conv2d_1/ActivityRegularizer/strided_slice/stack:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_1/ActivityRegularizer/CastCast3conv2d_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_1/ActivityRegularizer/truedivRealDiv5conv2d_1/ActivityRegularizer/PartitionedCall:output:0%conv2d_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_97324conv2d_2_97326*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_97323?
,conv2d_2/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
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
GPU2*0,1,2,3,4,5,6,7J 8? *8
f3R1
/__inference_conv2d_2_activity_regularizer_97204{
"conv2d_2/ActivityRegularizer/ShapeShape)conv2d_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_2/ActivityRegularizer/strided_sliceStridedSlice+conv2d_2/ActivityRegularizer/Shape:output:09conv2d_2/ActivityRegularizer/strided_slice/stack:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_2/ActivityRegularizer/CastCast3conv2d_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_2/ActivityRegularizer/truedivRealDiv5conv2d_2/ActivityRegularizer/PartitionedCall:output:0%conv2d_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
up_sampling2d/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *Q
fLRJ
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_97344?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_3_97358conv2d_3_97360*
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
GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_97357?
,conv2d_3/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
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
GPU2*0,1,2,3,4,5,6,7J 8? *8
f3R1
/__inference_conv2d_3_activity_regularizer_97236{
"conv2d_3/ActivityRegularizer/ShapeShape)conv2d_3/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_3/ActivityRegularizer/strided_sliceStridedSlice+conv2d_3/ActivityRegularizer/Shape:output:09conv2d_3/ActivityRegularizer/strided_slice/stack:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_3/ActivityRegularizer/CastCast3conv2d_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_3/ActivityRegularizer/truedivRealDiv5conv2d_3/ActivityRegularizer/PartitionedCall:output:0%conv2d_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_97383conv2d_4_97385*
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
GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_97382?
,conv2d_4/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
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
GPU2*0,1,2,3,4,5,6,7J 8? *8
f3R1
/__inference_conv2d_4_activity_regularizer_97249{
"conv2d_4/ActivityRegularizer/ShapeShape)conv2d_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_4/ActivityRegularizer/strided_sliceStridedSlice+conv2d_4/ActivityRegularizer/Shape:output:09conv2d_4/ActivityRegularizer/strided_slice/stack:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_4/ActivityRegularizer/CastCast3conv2d_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_4/ActivityRegularizer/truedivRealDiv5conv2d_4/ActivityRegularizer/PartitionedCall:output:0%conv2d_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????f

Identity_1Identity&conv2d/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_2Identity(conv2d_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_3Identity(conv2d_2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_4Identity(conv2d_3/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_5Identity(conv2d_4/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?d
?
@__inference_model_layer_call_and_return_conditional_losses_97792
input_1&
conv2d_97719:
conv2d_97721:(
conv2d_1_97733:
conv2d_1_97735:(
conv2d_2_97746:
conv2d_2_97748:(
conv2d_3_97760:
conv2d_3_97762:(
conv2d_4_97773:
conv2d_4_97775:
identity

identity_1

identity_2

identity_3

identity_4

identity_5??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_97719conv2d_97721*
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
GPU2*0,1,2,3,4,5,6,7J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_97267?
*conv2d/ActivityRegularizer/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU2*0,1,2,3,4,5,6,7J 8? *6
f1R/
-__inference_conv2d_activity_regularizer_97166w
 conv2d/ActivityRegularizer/ShapeShape'conv2d/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:x
.conv2d/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv2d/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv2d/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(conv2d/ActivityRegularizer/strided_sliceStridedSlice)conv2d/ActivityRegularizer/Shape:output:07conv2d/ActivityRegularizer/strided_slice/stack:output:09conv2d/ActivityRegularizer/strided_slice/stack_1:output:09conv2d/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d/ActivityRegularizer/CastCast1conv2d/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
"conv2d/ActivityRegularizer/truedivRealDiv3conv2d/ActivityRegularizer/PartitionedCall:output:0#conv2d/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
!average_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_97285?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0conv2d_1_97733conv2d_1_97735*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_97298?
,conv2d_1/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU2*0,1,2,3,4,5,6,7J 8? *8
f3R1
/__inference_conv2d_1_activity_regularizer_97191{
"conv2d_1/ActivityRegularizer/ShapeShape)conv2d_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_1/ActivityRegularizer/strided_sliceStridedSlice+conv2d_1/ActivityRegularizer/Shape:output:09conv2d_1/ActivityRegularizer/strided_slice/stack:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_1/ActivityRegularizer/CastCast3conv2d_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_1/ActivityRegularizer/truedivRealDiv5conv2d_1/ActivityRegularizer/PartitionedCall:output:0%conv2d_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_97746conv2d_2_97748*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_97323?
,conv2d_2/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
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
GPU2*0,1,2,3,4,5,6,7J 8? *8
f3R1
/__inference_conv2d_2_activity_regularizer_97204{
"conv2d_2/ActivityRegularizer/ShapeShape)conv2d_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_2/ActivityRegularizer/strided_sliceStridedSlice+conv2d_2/ActivityRegularizer/Shape:output:09conv2d_2/ActivityRegularizer/strided_slice/stack:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_2/ActivityRegularizer/CastCast3conv2d_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_2/ActivityRegularizer/truedivRealDiv5conv2d_2/ActivityRegularizer/PartitionedCall:output:0%conv2d_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
up_sampling2d/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *Q
fLRJ
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_97344?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_3_97760conv2d_3_97762*
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
GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_97357?
,conv2d_3/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
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
GPU2*0,1,2,3,4,5,6,7J 8? *8
f3R1
/__inference_conv2d_3_activity_regularizer_97236{
"conv2d_3/ActivityRegularizer/ShapeShape)conv2d_3/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_3/ActivityRegularizer/strided_sliceStridedSlice+conv2d_3/ActivityRegularizer/Shape:output:09conv2d_3/ActivityRegularizer/strided_slice/stack:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_3/ActivityRegularizer/CastCast3conv2d_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_3/ActivityRegularizer/truedivRealDiv5conv2d_3/ActivityRegularizer/PartitionedCall:output:0%conv2d_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_97773conv2d_4_97775*
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
GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_97382?
,conv2d_4/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
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
GPU2*0,1,2,3,4,5,6,7J 8? *8
f3R1
/__inference_conv2d_4_activity_regularizer_97249{
"conv2d_4/ActivityRegularizer/ShapeShape)conv2d_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_4/ActivityRegularizer/strided_sliceStridedSlice+conv2d_4/ActivityRegularizer/Shape:output:09conv2d_4/ActivityRegularizer/strided_slice/stack:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_4/ActivityRegularizer/CastCast3conv2d_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_4/ActivityRegularizer/truedivRealDiv5conv2d_4/ActivityRegularizer/PartitionedCall:output:0%conv2d_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????f

Identity_1Identity&conv2d/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_2Identity(conv2d_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_3Identity(conv2d_2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_4Identity(conv2d_3/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_5Identity(conv2d_4/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?d
?
@__inference_model_layer_call_and_return_conditional_losses_97868
input_1&
conv2d_97795:
conv2d_97797:(
conv2d_1_97809:
conv2d_1_97811:(
conv2d_2_97822:
conv2d_2_97824:(
conv2d_3_97836:
conv2d_3_97838:(
conv2d_4_97849:
conv2d_4_97851:
identity

identity_1

identity_2

identity_3

identity_4

identity_5??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_97795conv2d_97797*
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
GPU2*0,1,2,3,4,5,6,7J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_97267?
*conv2d/ActivityRegularizer/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU2*0,1,2,3,4,5,6,7J 8? *6
f1R/
-__inference_conv2d_activity_regularizer_97166w
 conv2d/ActivityRegularizer/ShapeShape'conv2d/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:x
.conv2d/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0conv2d/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0conv2d/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(conv2d/ActivityRegularizer/strided_sliceStridedSlice)conv2d/ActivityRegularizer/Shape:output:07conv2d/ActivityRegularizer/strided_slice/stack:output:09conv2d/ActivityRegularizer/strided_slice/stack_1:output:09conv2d/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d/ActivityRegularizer/CastCast1conv2d/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
"conv2d/ActivityRegularizer/truedivRealDiv3conv2d/ActivityRegularizer/PartitionedCall:output:0#conv2d/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
!average_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_97285?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:0conv2d_1_97809conv2d_1_97811*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_97298?
,conv2d_1/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU2*0,1,2,3,4,5,6,7J 8? *8
f3R1
/__inference_conv2d_1_activity_regularizer_97191{
"conv2d_1/ActivityRegularizer/ShapeShape)conv2d_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_1/ActivityRegularizer/strided_sliceStridedSlice+conv2d_1/ActivityRegularizer/Shape:output:09conv2d_1/ActivityRegularizer/strided_slice/stack:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_1/ActivityRegularizer/CastCast3conv2d_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_1/ActivityRegularizer/truedivRealDiv5conv2d_1/ActivityRegularizer/PartitionedCall:output:0%conv2d_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_97822conv2d_2_97824*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_97323?
,conv2d_2/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
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
GPU2*0,1,2,3,4,5,6,7J 8? *8
f3R1
/__inference_conv2d_2_activity_regularizer_97204{
"conv2d_2/ActivityRegularizer/ShapeShape)conv2d_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_2/ActivityRegularizer/strided_sliceStridedSlice+conv2d_2/ActivityRegularizer/Shape:output:09conv2d_2/ActivityRegularizer/strided_slice/stack:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_2/ActivityRegularizer/CastCast3conv2d_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_2/ActivityRegularizer/truedivRealDiv5conv2d_2/ActivityRegularizer/PartitionedCall:output:0%conv2d_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
up_sampling2d/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *Q
fLRJ
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_97344?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_3_97836conv2d_3_97838*
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
GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_97357?
,conv2d_3/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
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
GPU2*0,1,2,3,4,5,6,7J 8? *8
f3R1
/__inference_conv2d_3_activity_regularizer_97236{
"conv2d_3/ActivityRegularizer/ShapeShape)conv2d_3/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_3/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_3/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_3/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_3/ActivityRegularizer/strided_sliceStridedSlice+conv2d_3/ActivityRegularizer/Shape:output:09conv2d_3/ActivityRegularizer/strided_slice/stack:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_3/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_3/ActivityRegularizer/CastCast3conv2d_3/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_3/ActivityRegularizer/truedivRealDiv5conv2d_3/ActivityRegularizer/PartitionedCall:output:0%conv2d_3/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_97849conv2d_4_97851*
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
GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_97382?
,conv2d_4/ActivityRegularizer/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
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
GPU2*0,1,2,3,4,5,6,7J 8? *8
f3R1
/__inference_conv2d_4_activity_regularizer_97249{
"conv2d_4/ActivityRegularizer/ShapeShape)conv2d_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0conv2d_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_4/ActivityRegularizer/strided_sliceStridedSlice+conv2d_4/ActivityRegularizer/Shape:output:09conv2d_4/ActivityRegularizer/strided_slice/stack:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_1:output:0;conv2d_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!conv2d_4/ActivityRegularizer/CastCast3conv2d_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$conv2d_4/ActivityRegularizer/truedivRealDiv5conv2d_4/ActivityRegularizer/PartitionedCall:output:0%conv2d_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: ?
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????f

Identity_1Identity&conv2d/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_2Identity(conv2d_1/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_3Identity(conv2d_2/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_4Identity(conv2d_3/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: h

Identity_5Identity(conv2d_4/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
I
-__inference_up_sampling2d_layer_call_fn_98264

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *Q
fLRJ
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_97220?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
A__inference_conv2d_layer_call_and_return_conditional_losses_98340

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
?
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_97298

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
:?????????*
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
:?????????^
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????b
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_conv2d_2_layer_call_and_return_all_conditional_losses_98259

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
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_97323?
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
GPU2*0,1,2,3,4,5,6,7J 8? *8
f3R1
/__inference_conv2d_2_activity_regularizer_97204w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????X

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
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
/__inference_conv2d_3_activity_regularizer_97236
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
?
h
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_98214

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_98373

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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_98362

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
:?????????*
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
:?????????^
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????b
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
M
1__inference_average_pooling2d_layer_call_fn_98204

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_97175?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_conv2d_3_layer_call_and_return_all_conditional_losses_98309

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
GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_97357?
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
GPU2*0,1,2,3,4,5,6,7J 8? *8
f3R1
/__inference_conv2d_3_activity_regularizer_97236w
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
?
d
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_98289

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"      X
Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Q
mulMulConst:output:0Const_1:output:0*
T0*
_output_shapes
:?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(}
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
#__inference_signature_wrapper_97901
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *)
f$R"
 __inference__wrapped_model_97153w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
d
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_97220

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_98219

inputs
identity?
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
`
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_98651
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias:<
"assignvariableop_2_conv2d_1_kernel:.
 assignvariableop_3_conv2d_1_bias:<
"assignvariableop_4_conv2d_2_kernel:.
 assignvariableop_5_conv2d_2_bias:<
"assignvariableop_6_conv2d_3_kernel:.
 assignvariableop_7_conv2d_3_bias:<
"assignvariableop_8_conv2d_4_kernel:.
 assignvariableop_9_conv2d_4_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: #
assignvariableop_15_total: #
assignvariableop_16_count: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: B
(assignvariableop_19_adam_conv2d_kernel_m:4
&assignvariableop_20_adam_conv2d_bias_m:D
*assignvariableop_21_adam_conv2d_1_kernel_m:6
(assignvariableop_22_adam_conv2d_1_bias_m:D
*assignvariableop_23_adam_conv2d_2_kernel_m:6
(assignvariableop_24_adam_conv2d_2_bias_m:D
*assignvariableop_25_adam_conv2d_3_kernel_m:6
(assignvariableop_26_adam_conv2d_3_bias_m:D
*assignvariableop_27_adam_conv2d_4_kernel_m:6
(assignvariableop_28_adam_conv2d_4_bias_m:B
(assignvariableop_29_adam_conv2d_kernel_v:4
&assignvariableop_30_adam_conv2d_bias_v:D
*assignvariableop_31_adam_conv2d_1_kernel_v:6
(assignvariableop_32_adam_conv2d_1_bias_v:D
*assignvariableop_33_adam_conv2d_2_kernel_v:6
(assignvariableop_34_adam_conv2d_2_bias_v:D
*assignvariableop_35_adam_conv2d_3_kernel_v:6
(assignvariableop_36_adam_conv2d_3_bias_v:D
*assignvariableop_37_adam_conv2d_4_kernel_v:6
(assignvariableop_38_adam_conv2d_4_bias_v:
identity_40??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*?
value?B?(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_conv2d_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_conv2d_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv2d_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv2d_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv2d_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv2d_2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_4_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_4_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_conv2d_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_conv2d_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv2d_1_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv2d_1_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv2d_2_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv2d_2_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv2d_3_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv2d_3_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_4_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv2d_4_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_40IdentityIdentity_39:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_40Identity_40:output:0*c
_input_shapesR
P: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
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
?
?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_98384

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
(__inference_conv2d_2_layer_call_fn_98248

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
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_97323w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
%__inference_model_layer_call_fn_97716
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout

2*
_collective_manager_ids
 *9
_output_shapes'
%:?????????: : : : : *,
_read_only_resource_inputs

	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_97658w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
(__inference_conv2d_4_layer_call_fn_98318

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
GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_97382w
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
?
M
1__inference_average_pooling2d_layer_call_fn_98209

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_97285h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_conv2d_3_layer_call_fn_98298

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
GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_97357w
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
?
G__inference_conv2d_4_layer_call_and_return_all_conditional_losses_98329

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
GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_97382?
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
GPU2*0,1,2,3,4,5,6,7J 8? *8
f3R1
/__inference_conv2d_4_activity_regularizer_97249w
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_98351

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
:?????????*
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
:?????????^
SigmoidSigmoidBiasAdd:output:0*
T0*/
_output_shapes
:?????????b
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
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
input_18
serving_default_input_1:0?????????D
conv2d_48
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_network
6
_init_input_shape"
_tf_keras_input_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
6iter

7beta_1

8beta_2
	9decay
:learning_ratemnmompmq mr!ms*mt+mu0mv1mwvxvyvzv{ v|!v}*v~+v0v?1v?"
	optimizer
f
0
1
2
3
 4
!5
*6
+7
08
19"
trackable_list_wrapper
f
0
1
2
3
 4
!5
*6
+7
08
19"
trackable_list_wrapper
 "
trackable_list_wrapper
?
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics

	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
':%2conv2d/kernel
:2conv2d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_1/kernel
:2conv2d_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_2/kernel
:2conv2d_2/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
"	variables
#trainable_variables
$regularization_losses
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
&	variables
'trainable_variables
(regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_3/kernel
:2conv2d_3/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
,	variables
-trainable_variables
.regularization_losses
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_4/kernel
:2conv2d_4/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
2	variables
3trainable_variables
4regularization_losses
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
.
c0
d1"
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
	etotal
	fcount
g	variables
h	keras_api"
_tf_keras_metric
^
	itotal
	jcount
k
_fn_kwargs
l	variables
m	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
e0
f1"
trackable_list_wrapper
-
g	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
i0
j1"
trackable_list_wrapper
-
l	variables"
_generic_user_object
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
.:,2Adam/conv2d_1/kernel/m
 :2Adam/conv2d_1/bias/m
.:,2Adam/conv2d_2/kernel/m
 :2Adam/conv2d_2/bias/m
.:,2Adam/conv2d_3/kernel/m
 :2Adam/conv2d_3/bias/m
.:,2Adam/conv2d_4/kernel/m
 :2Adam/conv2d_4/bias/m
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
.:,2Adam/conv2d_1/kernel/v
 :2Adam/conv2d_1/bias/v
.:,2Adam/conv2d_2/kernel/v
 :2Adam/conv2d_2/bias/v
.:,2Adam/conv2d_3/kernel/v
 :2Adam/conv2d_3/bias/v
.:,2Adam/conv2d_4/kernel/v
 :2Adam/conv2d_4/bias/v
?2?
%__inference_model_layer_call_fn_97430
%__inference_model_layer_call_fn_97931
%__inference_model_layer_call_fn_97961
%__inference_model_layer_call_fn_97716?
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
@__inference_model_layer_call_and_return_conditional_losses_98070
@__inference_model_layer_call_and_return_conditional_losses_98179
@__inference_model_layer_call_and_return_conditional_losses_97792
@__inference_model_layer_call_and_return_conditional_losses_97868?
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
 __inference__wrapped_model_97153input_1"?
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
&__inference_conv2d_layer_call_fn_98188?
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
E__inference_conv2d_layer_call_and_return_all_conditional_losses_98199?
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
?2?
1__inference_average_pooling2d_layer_call_fn_98204
1__inference_average_pooling2d_layer_call_fn_98209?
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
?2?
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_98214
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_98219?
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
(__inference_conv2d_1_layer_call_fn_98228?
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
G__inference_conv2d_1_layer_call_and_return_all_conditional_losses_98239?
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
(__inference_conv2d_2_layer_call_fn_98248?
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
G__inference_conv2d_2_layer_call_and_return_all_conditional_losses_98259?
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
?2?
-__inference_up_sampling2d_layer_call_fn_98264
-__inference_up_sampling2d_layer_call_fn_98269?
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
?2?
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_98281
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_98289?
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
(__inference_conv2d_3_layer_call_fn_98298?
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
G__inference_conv2d_3_layer_call_and_return_all_conditional_losses_98309?
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
(__inference_conv2d_4_layer_call_fn_98318?
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
G__inference_conv2d_4_layer_call_and_return_all_conditional_losses_98329?
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
#__inference_signature_wrapper_97901input_1"?
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
-__inference_conv2d_activity_regularizer_97166?
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
A__inference_conv2d_layer_call_and_return_conditional_losses_98340?
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
/__inference_conv2d_1_activity_regularizer_97191?
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_98351?
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
/__inference_conv2d_2_activity_regularizer_97204?
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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_98362?
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
/__inference_conv2d_3_activity_regularizer_97236?
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
C__inference_conv2d_3_layer_call_and_return_conditional_losses_98373?
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
/__inference_conv2d_4_activity_regularizer_97249?
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
C__inference_conv2d_4_layer_call_and_return_conditional_losses_98384?
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
 __inference__wrapped_model_97153?
 !*+018?5
.?+
)?&
input_1?????????
? ";?8
6
conv2d_4*?'
conv2d_4??????????
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_98214?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_98219h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
1__inference_average_pooling2d_layer_call_fn_98204?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
1__inference_average_pooling2d_layer_call_fn_98209[7?4
-?*
(?%
inputs?????????
? " ??????????Y
/__inference_conv2d_1_activity_regularizer_97191&?
?
?	
x
? "? ?
G__inference_conv2d_1_layer_call_and_return_all_conditional_losses_98239z7?4
-?*
(?%
inputs?????????
? ";?8
#? 
0?????????
?
?	
1/0 ?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_98351l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
(__inference_conv2d_1_layer_call_fn_98228_7?4
-?*
(?%
inputs?????????
? " ??????????Y
/__inference_conv2d_2_activity_regularizer_97204&?
?
?	
x
? "? ?
G__inference_conv2d_2_layer_call_and_return_all_conditional_losses_98259z !7?4
-?*
(?%
inputs?????????
? ";?8
#? 
0?????????
?
?	
1/0 ?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_98362l !7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
(__inference_conv2d_2_layer_call_fn_98248_ !7?4
-?*
(?%
inputs?????????
? " ??????????Y
/__inference_conv2d_3_activity_regularizer_97236&?
?
?	
x
? "? ?
G__inference_conv2d_3_layer_call_and_return_all_conditional_losses_98309z*+7?4
-?*
(?%
inputs?????????
? ";?8
#? 
0?????????
?
?	
1/0 ?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_98373l*+7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
(__inference_conv2d_3_layer_call_fn_98298_*+7?4
-?*
(?%
inputs?????????
? " ??????????Y
/__inference_conv2d_4_activity_regularizer_97249&?
?
?	
x
? "? ?
G__inference_conv2d_4_layer_call_and_return_all_conditional_losses_98329z017?4
-?*
(?%
inputs?????????
? ";?8
#? 
0?????????
?
?	
1/0 ?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_98384l017?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
(__inference_conv2d_4_layer_call_fn_98318_017?4
-?*
(?%
inputs?????????
? " ??????????W
-__inference_conv2d_activity_regularizer_97166&?
?
?	
x
? "? ?
E__inference_conv2d_layer_call_and_return_all_conditional_losses_98199z7?4
-?*
(?%
inputs?????????
? ";?8
#? 
0?????????
?
?	
1/0 ?
A__inference_conv2d_layer_call_and_return_conditional_losses_98340l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
&__inference_conv2d_layer_call_fn_98188_7?4
-?*
(?%
inputs?????????
? " ???????????
@__inference_model_layer_call_and_return_conditional_losses_97792?
 !*+01@?=
6?3
)?&
input_1?????????
p 

 
? "s?p
#? 
0?????????
I?F
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 
?	
1/4 ?
@__inference_model_layer_call_and_return_conditional_losses_97868?
 !*+01@?=
6?3
)?&
input_1?????????
p

 
? "s?p
#? 
0?????????
I?F
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 
?	
1/4 ?
@__inference_model_layer_call_and_return_conditional_losses_98070?
 !*+01??<
5?2
(?%
inputs?????????
p 

 
? "s?p
#? 
0?????????
I?F
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 
?	
1/4 ?
@__inference_model_layer_call_and_return_conditional_losses_98179?
 !*+01??<
5?2
(?%
inputs?????????
p

 
? "s?p
#? 
0?????????
I?F
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 
?	
1/4 ?
%__inference_model_layer_call_fn_97430p
 !*+01@?=
6?3
)?&
input_1?????????
p 

 
? " ???????????
%__inference_model_layer_call_fn_97716p
 !*+01@?=
6?3
)?&
input_1?????????
p

 
? " ???????????
%__inference_model_layer_call_fn_97931o
 !*+01??<
5?2
(?%
inputs?????????
p 

 
? " ???????????
%__inference_model_layer_call_fn_97961o
 !*+01??<
5?2
(?%
inputs?????????
p

 
? " ???????????
#__inference_signature_wrapper_97901?
 !*+01C?@
? 
9?6
4
input_1)?&
input_1?????????";?8
6
conv2d_4*?'
conv2d_4??????????
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_98281?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_98289h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
-__inference_up_sampling2d_layer_call_fn_98264?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
-__inference_up_sampling2d_layer_call_fn_98269[7?4
-?*
(?%
inputs?????????
? " ??????????