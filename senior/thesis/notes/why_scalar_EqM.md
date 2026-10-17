# Why do scalar EqM? Why my specific implementation?

Basically, Yilun told me to do so:

<SLACK-HISTORY>

Matthew Krasnow  [9:00 AM]
I was wondering whether enforcing conservativeness could be useful. It would make the energy interpretation exact and might improve sampling stability / generalization away from the corruption trajectories seen during training. I would parameterize a scalar E(x) and train the gradient of E(x) against the existing EqM target.

Do you think this is worth investigating as well?
Yilun Du  [9:22 AM]
Hi Matthew actually do you want to just try to directly train a scalar E(x)?
[9:22 AM]In later projects, we found that to work pretty well
Matthew Krasnow  [11:35 AM]
Do you mean directly regressing a scalar E(x) onto an energy target for each corrupted sample, rather than training the gradient of E(x) to match the EqM vector-field target?
Yilun Du  [12:03 PM]
So directly regression a scalar energy using a neural network
[12:03 PM]And then targeting the gradient of E(x) to match the EqM field vector target

</SLACK-HISTORY>
