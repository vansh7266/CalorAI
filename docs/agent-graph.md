# CalorAI agent graph

Generated from the compiled LangGraph with `python scripts/render_graph.py`.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	ingest(ingest)
	load_context(load_context)
	vision_extract(vision_extract)
	agent(agent)
	tools(tools)
	__end__([<p>__end__</p>]):::last
	__start__ --> ingest;
	agent -.-> __end__;
	agent -.-> tools;
	ingest -.-> load_context;
	ingest -.-> vision_extract;
	load_context --> agent;
	tools --> agent;
	vision_extract --> agent;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```

```text
              +-----------+                 
              | __start__ |                 
              +-----------+                 
                     *                      
                     *                      
                     *                      
                +--------+                  
                | ingest |                  
                +--------+.                 
              ..           ..               
            ..               ..             
          ..                   ..           
+--------------+         +----------------+ 
| load_context |         | vision_extract | 
+--------------+         +----------------+ 
              **           **               
                **       **                 
                  **   **                   
                +-------+                   
                | agent |                   
                +-------+.                  
                .         .                 
              ..           ..               
             .               .              
      +---------+         +-------+         
      | __end__ |         | tools |         
      +---------+         +-------+         
```
