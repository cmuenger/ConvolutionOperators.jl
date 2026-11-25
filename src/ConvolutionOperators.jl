module ConvolutionOperators

using LinearAlgebra


abstract type AbstractConvOp end

function Base.size(x::AbstractConvOp) map(length, axes(x)) end
function Base.size(x::AbstractConvOp, i) size(x)[i] end
function Base.axes(x::AbstractConvOp) map(Base.OneTo, size(x)) end
function Base.axes(x::AbstractConvOp, i) axes(x)[i] end

const _NUM_THREADS = Threads.Atomic{Int}(0)   # 0: use Threads.nthreads()

"""
    set_num_threads(n)

Set the number of tasks that `convolve!` uses (as `BLAS.set_num_threads`): `n = 1` runs it
serially in the calling task, e.g. when the caller already parallelizes over many
convolutions; `n = 0` restores the default `Threads.nthreads()`. The setting is global for
the process.
"""
function set_num_threads(n::Integer)
    n >= 0 || throw(ArgumentError("the number of threads must be ≥ 0, got $n"))
    _NUM_THREADS[] = n
    return nothing
end

"""
    get_num_threads()

Number of tasks that `convolve!` uses, see `set_num_threads`.
"""
get_num_threads() = (n = _NUM_THREADS[]; n == 0 ? Threads.nthreads() : n)

function timeslice(Z,k)
    T = eltype(Z)
    Zk = zeros(T, size(Z)[1:2])
    ConvolutionOperators.timeslice!(Zk, Z, k)
end

include("companion.jl")

include("zeroconvop.jl")
include("denseconvop.jl")
include("bandedconvop.jl")
include("linearcombinations.jl")
include("leftrightmul.jl")
include("liftedconvop.jl")
include("truncatedconvop.jl")
include("matrixconvop.jl")


end
