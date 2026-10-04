This is a highly optimized inference library for serving qwen 3.8 flash next MoE model, based on llama cpp and custom code.
The problem is that the engine doesnt support concurrent requests properly currently.

The engine already uses shared ressources so that multiple setup.py model servers can be launched on the machine, however I also need concurrency for each server process itself at the token / step level.

So in order to get there I want:

STAGE 4

Continous batching would be a tedious rewrite of the core engine
(aka batching prefill and text generation in one GPU pass for multiple requests simulatonously)
So I propose a different architecture idea:

The server already supports multiple instances which can do prefill/text gen individually,
so I wonder if it would be doable to do multiple processes which split both stages and focus on one.

My idea is to have one prefill-only instance which does pre-filling and multiple text-gen endpoints which all use the shared prefill-instance and do text gen individually.

Requests may connect to the text-gen instances which then communicate with the prefill-instance.

The prefill instance is more of a backend artifact here, and doesnt accept direct requests from other clients than the text gen instances.

What I want at the end is that the text gen stage runs concurrently (at a token level - to get true concurrency / batching) 
without having to implement the complexities of continous coupled batching / coupling prefill and textgen on requests, this should be doable.

Note do not "pin" that to the current machine's limits. In the future it will receive more RAM to make more requests more feasible.

Summary: 

- Different server instances pinned to different GPU's to handle prefill / textgen stage separately
- The textgen stage itself should implement real batching of concurrent requests on the GPU (not just simple swap&park)

At the end, it should look like this for example:

PREFILL INSTANCE (handling prefill for all textgens)
^ TEXTGEN INSTANCE (serving 2 requests simulatonously)
^ TEXTGEN INSTANCE (serving 3 requests)


Also a proposed design change for the API:

Currently if there is no budget to park a conversation or different circumstances requests get rejected with an error.
That is bad, instead, those requests should be put on hold and be executed as soon as there are ressources free instead of cancelled.


IMPORTANT: 

Do NOT start the server! The machine doesnt have enough ressources free currently to host the model (as two running instances of it must stay active)