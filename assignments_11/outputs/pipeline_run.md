# Reflection

### Did the pipeline run cleanly on the first try? If not, what failed and how did you fix it?

Answer:
No, the pipeline did not run cleanly on the first try. It failed because of a code error in `confidences = probabilities.max(axis=1)` with `an exception: AxisError: axis 1 is out of bounds for array of dimension 1')`. I fixed the issue by removing that line and then reran the pipeline successfully.

I originally used this code because I misunderstood the requirement and thought I needed to get the highest probability from the model’s predictions and use it as the confidence level for the prediction. However, the confidence rate was actually meant to indicate whether it was a good day to run.

### What did the Prefect UI show? Did any tasks retry?

Answer:
The Prefect UI showed that a task had failed. When I clicked on the failed task, I could expand it to see the detailed logs of of the failed pipeline and the error message. No tasks retried because the pipeline stopped when the error occurred.

### Look at a few rows in weather_enriched. Do the LLM summaries seem accurate and useful? Pick one that stands out (positively or negatively) and explain why.

Answer:
After looking at a few rows in weather_enriched, some of the LLM summaries did not seem very focused. In many cases, the main recommendation was placed at the end of the sentence instead of being stated clearly at the beginning. It began with describing the weather conditions then suggestion to run or not. One thing that stood out negatively was that some months appeared to have weather conditions that were not too bad for running, but the LLM still recommended against running. For examples,in cooler temperature and moderate wind, the LLM suggested not good for running. This may be because the LLM is relying heavily on the ML model's predicted probabilities rather than considering the actual weather conditions on their own. I would make the summaries more directly explain how the weather factors and prediction led to the recommendation and put run or not before the weather conditions.

### What is one thing you would change or add if you were deploying this pipeline to run on a daily schedule — fetching the previous day's forecast each morning and enriching it automatically?

Answer:
One thing I would change is reducing the pipeline's latency as the amount of data increases. Currently, the pipeline makes many API calls, which makes it slow. I would add batching so that data can be fetched and processed more efficiently. For a daily schedule, I would also limit the API request to the previous day's forecast instead of repeatedly fetching unnecessary historical data. This would reduce the number of API calls and make the pipeline faster and more efficient.
