Okay, so for the chatboard.



Um, okay, sorry, for the, for the next step in this process, we are looking to create a guardrail, and this gutrail will be powered by the latest, uh, Singbird head explain and synthetic Singapore head speech. Classify classification data.



So using that most highest performing model, it will power our guardrail that will function within our chatboard application.



So powering this chatboard, I'm looking to use Gemini 3.one flash.



So Jemina 3.one flash seems to be quite uh okay with pumping out examples of Singaporean head speech.



And so the gutrail ideally should, Perhaps we can make it exist in a flow where there's one that protects the model's output.



So it doesn't allow the model to output any Singaporean head speech in response to a user.



So even if they prompt the model to ask it for that kind of output, or to really, like, rephrase what they say, or in any way sort of form like that, it should be a violation to the model.



And so the the LLM should therefore reply saying that you can't help with that.



As for the other flow, it could be that the user's input will trigger a flag from the guardrail in the model and then uh, It will be interesting if we show that it stops there.



But then at the same time, uh, I, I guess it's more like a system generated error rather than the LLM output, saying that, uh, you, it can't help with this question.



Or, or maybe the LLM output one will be a bit for more smarter chat response, whereas the input one would be like a block from the model itself.



Maybe that is a good system to demo, because I think the realistic use case, or at least how AI systems typically work nowadays for all of these additional guard rails on AI systems, is that they're typically just toggles or guard rails, and having an entire system just to classify Entire, even a LLM, like a large LLM, running inference on these cases may not be as efficient.



And based on the earlier test of accuracy, it may not even be that accurate.



So perhaps the alums job is not to actually classify, but to assist with the next text output from the chatboard in this case, or even other scenarios such as advising the content moderators of the types of violations that people are sending in or that exist on the system. Rather than using the LLM itself to classify something as hit or not.



So we are reframing the problem into the way that we can use the gutrail as a catch in the chatboard output or input.



So the framing is quite important where we need it to be in a chatboard flow.



So I'm looking to experiment with a few ways to demonstrate this.



Some kind of visibility of the backend processing or the decision boundary or like the, the, The, the kind of just to show the thinking process of when the model gets the input, how does it reach the gutrail and whether or not it's triggered?



I think that process will be good to visualise some kind of web that connects to the processing flow.



And then for the web itself, the checkboard itself, um, We can probably reference some GitHub ripple to easily have a chatboard with basic memory and like turn sequencing so that we can just show that if we give this, yeah, we put an API for the LM itself.



The chat will flow with back and forth and the chatboard. Can remember the history.



Something as simple as that just to demonstrate how the guardrail would function in a use case, for example.



Um, That would be good to see.

