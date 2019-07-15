include("./utils.jl")

using .Utils
using Knet: progress!, minibatch, adam, relu

EPOCH = 1
BATCHSIZE = 64
SENTENCELEN = 300
DROPOUT = 0.5
LR = 0.001
TRUNCATE = true

EMBEDDINGDIM = 100
HIDDENDIM = 20
OUTPUTDIM = 2

CNN_W1 = 5
CNN_W2 = 5
CNN_CX = 1
CNN_CY = 20

MODELTYPE = 1 # 0: MLP, 1: RNN, 2: CNN

if MODELTYPE == 0
    println("Training MLP...")
elseif MODELTYPE == 1
    println("Training RNN...")
else
    println("Training CNN...")
end

TRAIN_POS_PATH = joinpath(@__DIR__, "aclImdb/train/pos/")
TRAIN_NEG_PATH = joinpath(@__DIR__, "aclImdb/train/neg/")
TEST_POS_PATH = joinpath(@__DIR__, "aclImdb/test/pos/")
TEST_NEG_PATH = joinpath(@__DIR__, "aclImdb/test/neg/")

println("Read Training Data...")

xtrn_pos, ytrn_pos = readfolder(TRAIN_POS_PATH; class=1)
xtrn_neg, ytrn_neg = readfolder(TRAIN_NEG_PATH; class=2)

xtrn = vcat(xtrn_pos, xtrn_neg)
ytrn = vcat(ytrn_pos, ytrn_neg)

println("Read Test Data...")

xtst_pos, ytst_pos = readfolder(TEST_POS_PATH; class=1)
xtst_neg, ytst_neg = readfolder(TEST_NEG_PATH; class=2)

xtst = vcat(xtst_pos, xtst_neg)
ytst = vcat(ytst_pos, ytst_neg)

println("Shuffle...")

xtrn, ytrn = shuffledata(xtrn, ytrn)

println("Build vocabulary...")

voc = vocabulary(frequencies(skipwords(map(doc->map(lowercase, doc), xtrn))), 10)
len_voc = length(voc)

println("Doc2ids and add padding...")


if MODELTYPE == 0
    pos = "post"
    TRUNCATE = false
else
    pos = "pre"
end

xtrn_ids = addpadding!(doc2ids(xtrn, voc), pos=pos)
xtst_ids = addpadding!(doc2ids(xtst, voc), pos=pos)

if TRUNCATE
    println("Truncate sentences...")

    xtrn_ids = truncate!(xtrn_ids, SENTENCELEN, pos=pos)
    xtst_ids = truncate!(xtst_ids, SENTENCELEN, pos=pos)
end

dtrn = minibatch(xtrn_ids, ytrn, BATCHSIZE; shuffle=true)
dtst = minibatch(xtst_ids, ytst, BATCHSIZE; shuffle=true)

println("Build model...")

if MODELTYPE == 0
    model = MLPClassifier(len_voc, EMBEDDINGDIM, HIDDENDIM, OUTPUTDIM, relu; pdrop=0.5)
elseif MODELTYPE == 1
    model = RNNClassifier(len_voc, EMBEDDINGDIM, HIDDENDIM, OUTPUTDIM; pdrop=DROPOUT)
elseif MODELTYPE == 2
    dtrn = minibatch(xtrn_ids, ytrn, BATCHSIZE; shuffle=true)
    dtst = minibatch(xtst_ids, ytst, BATCHSIZE; shuffle=true)

    HIDDENDIM = Int((EMBEDDINGDIM - 4) / 2 * (SENTENCELEN - 4) / 2 * CNN_CY)
    model = CNNClassifier(len_voc, EMBEDDINGDIM, CNN_W1, CNN_W2,
                          CNN_CX, CNN_CY, HIDDENDIM, OUTPUTDIM,
                          relu; pdrop=DROPOUT)
else
    model = MLPClassifier(len_voc, EMBEDDINGDIM, HIDDENDIM, OUTPUTDIM, relu; pdrop=DROPOUT)
end

println("Training...")
progress!(adam(model, repeat(dtrn, EPOCH), lr=LR))

println("Train Accuracy:")
println(accuracy(model, dtrn))

println("Test Accuracy:")
println(accuracy(model, dtst))

println("Finish!")

