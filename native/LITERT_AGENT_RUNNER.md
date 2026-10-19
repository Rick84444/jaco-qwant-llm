## Sixteen bounded local LLM review missions

The mission inventory in native/litert_agent_missions.json contains exactly
sixteen independent, source-grounded code reviews for the final ACFQ-LiteRT
attempt. The native/run_litert_agents.py script reads only specified files,
passes excerpts to local Gemma 4 E2B at 127.0.0.1:9379, sequentially creates
one independent model context per specialist assignment and stores markdown
reports in ~/jaco-litert-agent-reports outside the checkout.

It does not spawn 16 parallel OS worker processes; it does not run shell
commands suggested by the model, modify LiteRT, merge changes, or claim that
LLM-generated reviews are proven. It can be run with:

    python3 native/run_litert_agents.py --dry-run
    python3 native/run_litert_agents.py --run

Only the second command actually executes the sixteen independent local
LLM reviews, provided the Gemma HTTP endpoint is healthy. It does not start
on its own. Each report includes prompt and source hashes; the manifest
records requests, failures and file paths. Reruns skip existing reports
unless --force is supplied.

To claim true integration: independently verify the original TFLite operator
is linked in a rebuilt isolated LiteRT, then observe real Gemma decoder
invocation in that process. These reviews alone prove neither integration
nor acceleration.
