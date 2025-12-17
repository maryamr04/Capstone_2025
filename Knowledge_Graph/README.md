# Knowledge Graph and Attention Matrix Generation


Step by step for setting up your Neo4j instance:

1. Go to this [link](https://login.neo4j.com/u/login/identifier?state=hKFo2SBUb2lmSkZTR2tYZ0VtcGFoME9rTzhFdWh6Mk1zZnRvTKFur3VuaXZlcnNhbC1sb2dpbqN0aWTZIEFBQ0VVaVZaVEJhTW9DVWxWbFNyczVJbXMybG9VREJGo2NpZNkgRXZ2MmNjWFBjOHVPeGV3bzBJalkyMFlJckg3VmtKVzk)
2. Create an account by logging in with Google (or your preferred method)
3. This should give you ownership of a free, running instance
4. Copy down the URI (which will start something like this `neo4j+s://`), username and password for your instance and input it into the code cells where it says:
   
`# neo4j connection`

`URI = "YOUR_URI"`

`AUTH = ("USERNAME", "PASSWORD")`

`driver = GraphDatabase.driver(URI, auth=AUTH)`

5. If you have not used the instance in a while, be sure to go the same link above, log in, and make sure your instance was not paused. If it was paused, get it running again before you try to run the code.

Step by step for uploading documents into Graph Builder:

1. Go to this [link](https://llm-graph-builder.neo4jlabs.com/readonly)
2. Click the button in the upper right hand corner to connect your running instance to the graph builder (you will need the same credentials as above)
3. Once you are connected, in the bottom left hand corner, there is a dropdown menu of LLM options for you to generate the graph based on - Choose your preferred model
4. Go to the upload tab in the far left hand of the screen and choose which documents to generate a graph for (click the **Generate Graph** button in the bottom right once you are finished uploading)
6. Once the graphs are generated, you will then be able to click the **Preview Graph** button to see what the builder has generated

7. ## Running the Code

8. The code was written in a Google Colab environment with the appropriate install and import statements used throughout the notebook. Running these cells should get you the functionality you need.

