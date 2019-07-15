module Utils

using Random: shuffle
using Statistics: mean
using Knet: Knet, AutoGrad, Param, @diff, value, params, grad, progress, progress!, relu, load, save, minibatch, Data, dropout, conv4, RNN, pool, mat

export
    MLPClassifier,
    RNNClassifier,
    BiRNNClassifier,
    CNNClassifier,
    EmbedLayer,
    Layer,
    Chain,
    readtext,
    readfolder,
    unzip,
    shuffledata,
    tokenize,
    skipwords,
    frequencies,
    vocabulary,
    max_sentence,
    addpadding!,
    truncate!,
    doc2ids,
    accuracy,
    zeroone,
    convert2d,
    nll,
    sgdupdate!,
    sgd

struct EmbedLayer; w; b; end


EmbedLayer(numwords::Int, embed::Int, scale=0.01) = EmbedLayer(
    Param(scale * randn(embed, numwords)))

function (l::EmbedLayer)(input)
    embed = l.input[:, permutedims(hcat(input...))]
    output = sum(embed, dims=3)[:,:,end]
end
    
(m::EmbedLayer)(x, y) = nll(m(x), y)


struct Layer; w; b; f; end

Layer(i::Int, o::Int, scale=0.01, f=relu) = Layer(
    Param(scale * randn(o, i)), Param(zeros(o)), f)

(m::Layer)(x) = m.f.(m.w * x .+ m.b)
(m::Layer)(x, y) = nll(m(x), y)

struct Chain
    layers
    Chain(layers...) = new(layers)
end

(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain)(x, y) = nll(c(x), y)


struct MLPClassifier; input; hidden; b1; output; b2; f; pdrop; end

MLPClassifier(input::Int, embed::Int, hidden::Int, output::Int, f=relu; pdrop=0, scale=0.01) =
    MLPClassifier(Param(randn(embed, input) * scale),
                  Param(randn(hidden, embed) * scale),
                  Param(zeros(hidden)),
                  Param(randn(output, hidden) * scale),
                  Param(zeros(output)),
                  f,
                  pdrop)

function (c::MLPClassifier)(input)
    embed = c.input[:, permutedims(hcat(input...))]
    hiddeninput = sum(embed, dims=3)[:,:,end]
    hiddeninput = dropout(hiddeninput, c.pdrop)
    hiddenoutput = c.f.(c.hidden * hiddeninput .+ c.b1)
    hiddenoutput = dropout(hiddenoutput, c.pdrop)
    outputlayer = c.output * hiddenoutput .+ c.b2
    
    return outputlayer
end

(c::MLPClassifier)(input,output) = nll(c(input), output)

struct RNNClassifier; input; rnn; output; b; pdrop; end

RNNClassifier(input::Int, embed::Int, hidden::Int, output::Int; pdrop=0, scale=0.01) =
    RNNClassifier(Param(Float32.(randn(embed, input) * scale)),
                  RNN(embed, hidden, rnnType=:gru),
                  Param(randn(output, hidden) * scale),
                  Param(zeros(output)),
                  pdrop)

function (c::RNNClassifier)(input)
    embed = c.input[:, permutedims(hcat(input...))]
    embed = dropout(embed, c.pdrop)
    hiddenoutput = c.rnn(embed)
    hiddenoutput = dropout(hiddenoutput, c.pdrop)

    return c.output * hiddenoutput[:,:,end] .+ c.b 
end

(c::RNNClassifier)(input,output) = nll(c(input), output)

struct BiRNNClassifier; input; rnn; output; b; pdrop; end

BiRNNClassifier(input::Int, embed::Int, hidden::Int, output::Int; pdrop=0, scale=0.01, ) =
    BiRNNClassifier(Param(Float32.(randn(embed, input) * scale)),
                  RNN(embed, hidden, rnnType=:gru, bidirectional=true),
                  Param(randn(output, 2 * hidden) * scale),
                  Param(zeros(output)),
                  pdrop)

function (c::BiRNNClassifier)(input)
    embed = c.input[:, permutedims(hcat(input...))]
    embed = dropout(embed, c.pdrop)
    hiddenoutput = c.rnn(embed)
    hiddenoutput = dropout(hiddenoutput, c.pdrop)

    return c.output * hiddenoutput[:,:,end] .+ c.b 
end

(c::BiRNNClassifier)(input,output) = nll(c(input), output)


struct CNNClassifier; input; w; b1; output; b2; f; pdrop; end

CNNClassifier(input::Int, embed::Int, w1::Int, w2::Int, cx::Int, cy::Int, hidden::Int, output::Int, f=relu; pdrop=0, scale=0.01) =
    CNNClassifier(Param(randn(embed, input) * scale),
                   Param(randn(w1, w2, cx, cy) * scale),
                   Param(zeros(1, 1, cy, 1)),
                   Param(randn(output, hidden) * scale),
                   Param(zeros(output)),
                   f,
                   pdrop)

function (c::CNNClassifier)(input)
    embed = c.input[:, permutedims(hcat(input...), [1, 2])]
    embed = dropout(embed, c.pdrop)
    cnnoutput = c.f.(pool(conv4(c.w, reshape(
        embed, (size(embed,1), size(embed, 2), 1, size(embed, 3)))) .+ c.b1))
    return c.output * mat(cnnoutput) .+ c.b2
end

(c::CNNClassifier)(input,output) = nll(c(input), output)


struct Conv; w; b; f; p; end

(c::Conv)(x) = c.f.(pool(conv4(c.w, dropout(x, c.p)) .+ c.b))

Conv(w1::Int, w2::Int, cx::Int, cy::Int, f=relu; pdrop=0, scale=0.01) =
    Conv(Param(randn(w1, w2, cx, cy) * scale), zeros(1, 1, cy, 1), f, pdrop)

struct Dense; w; b; f; p; end

(d::Dense)(x) = d.f.(d.w * mat(dropout(x, d.p)) .+ d.b)

Dense(i::Int, o::Int, f=relu; pdrop=0) =
    Dense(Param(randn(o,i) * scale),
          Param(zeros(o)),
          f,
          pdrop)


function readtext(filename)
    file = open(filename, "r")
    text = read(file, String)
    close(file)
    text
end

function readfolder(path; class=0)
    files = readdir(path)
    docs = [tokenize(readtext(path * file)) for file in files]
    outputs = [class for i in docs]
    docs, outputs
end

unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

shuffledata(x, y) = unzip(shuffle(collect(zip(x, y))))

function tokenize(text)
    punctuations = ['.', '?', '!', ';', '(', ')']
    text = replace(text, "<br />" => "")

    for p in punctuations
        text = replace(text, "$p" => " $p")
    end

    tokenized = split(text)
end


function skipwords(x)
    stopwords = ["this", "is", "a", "an", "the",
                 '.', '?', '!', ';']

    removed = [[word for word in doc if !in(word, stopwords)]
               for doc in x]
end


function frequencies(x)
    freqs = Dict{String, Integer}()

    for doc in x
        for word in doc
            freqs[word] = 1 + get(freqs, word, 0)
        end
    end
    freqs
end
    
function vocabulary(freqs, threshold=1)
    vocabulary = Dict{String, Integer}()
    vocabulary["<pad>"] = 1
    pid = 2

    frequency_order = sort([(freqs[word], word)
                           for word in keys(freqs)], rev=true)

    for (f, w) in frequency_order
        if f >= threshold
            vocabulary[w] = pid
            pid += 1
        end
    end

    vocabulary["<unk>"] = pid
    vocabulary
end

function maxsentence(x)
    maximum(length(doc) for doc in x)
end
    
function addpadding!(x; pos="post")
    max_length = maxsentence(x)

    for doc in x
        while length(doc) < max_length
            if pos == "post"
                push!(doc, 1)
            else
                pushfirst!(doc, 1)
            end
        end
    end
    x
end

function truncate!(x, th=300; pos="post")
    for doc in x
        while length(doc) > th
            if pos == "post"
                pop!(doc)
            else
                popfirst!(doc)
            end
        end
    end
    x
end
    
function doc2ids(x, voc)
    x = [map(lowercase, doc) for doc in x]
    ids = [map(x -> haskey(voc, x) ? voc[x] : voc["<unk>"], doc)
           for doc in x]
end

function convert2d(x)
    permutedims(reshape(hcat(x...), (length(x[1]), length(x))))
end

function nll(scores, y)
    expscores = exp.(scores)
    probabilities = expscores ./ sum(expscores, dims=1)
    answerprobs = (probabilities[y[i],i] for i in 1:length(y))
    mean(-log.(answerprobs))
end

function sgdupdate!(func, args; lr=0.1)
    fval = @diff func(args...)
    for param in params(fval)
        ∇param = grad(fval, param)
        param .-= lr * ∇param
    end
    return value(fval)
end

sgd(func, data; lr=0.1) = 
    (sgdupdate!(func, args; lr=lr) for args in data)

accuracy(model, x, y) = mean(y' .== map(i->i[1], findmax(Array(model(x)),dims=1)[2]))

accuracy(model, data) = mean(accuracy(model,x,y) for (x,y) in data)

zeroone(x...) = 1 - accuracy(x...)

end
