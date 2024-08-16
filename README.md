# envisim_samplr
Provides design-based sampling methods, with a focus on spatially balanced and balanced sampling
designs.

> "everything is related to everything else, but near things are more related than distant things"
>
> &mdash; Tobler's first law of geography, Waldo Tobler

**Balanced sampling** utilizes auxilliary information in order to obtain a sample where
the Horvitz-Thompson (HT) estimator of the total of the auxilliary information equals the population
total of the auxilliaries.
This may be very efficient (yield relatively low variance) if there is a linear relationship between
the auxilliaries and the variable of interest.[^1]

**Spatially balanced sampling** uses auxilliary information in order to obtain a sample that is
well-spread in auxilliary space, as well as being balanced.
The samples can then be seen as a miniature version of the population.
This generally yields low variances for the variable of interest, if there is a general relationship
between the auxilliaries and the variables of interest.[^2]

[^1]: Grafström, A., & Tillé, Y. (2013).
Doubly balanced spatial sampling with spreading and restitution of auxiliary totals.
*Environmetrics*, 24(2), 120-131.
<https://doi.org/10.1002/env.2194>

[^2]: Grafström, A., & Schelin, L. (2014).
How to select representative samples.
*Scandinavian Journal of Statistics*, 41(2), 277-290.
<https://doi.org/10.1111/sjos.12016>

## Links
- [Envisim](https://envisim.se)

