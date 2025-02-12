# Covvfit: Variant fitness estimates from wastewater data

*Covvfit* is a framework for estimating relative growth advantages of different variants from deconvolved wastewater samples.
It consists of command line tools, which can be included in the existing workflows and a Python package, which can be used to quickly develop custom solutions and extensions.




## FAQ

**How do I run it on my data?**

We recommend to start using *Covvfit* as a command line tool, with the tutorial available [here](cli.md). 

**What data does *Covvfit* use?**

*Covvfit* uses deconvolved wastewater data, accepting relative abundances of different variants measured at different locations and times.
Tools such as [LolliPop](https://github.com/cbg-ethz/LolliPop) or [Freyja](https://github.com/andersen-lab/Freyja/) can be used to deconvolve wastewater data. 

**Can *Covvfit* predict emergence of new variants?**

No, *Covvfit* explicitly assumes that no new variants emerge and its predictions are unlikely to hold on longer timescales.
The underlying model also cannot take into account changes in the transmission dynamics or immune response, so that it cannot predict the effects of vaccination programs or lockdowns.  

**How can I contact the developers?**

In case you find a bug, want to ask about integrating *Covvfit* into your pipeline, or have any other feedback, we would love to hear it via our [issue tracker](https://github.com/cbg-ethz/covvfit/issues)!
In this manner, other users can also benefit from your insights.

**Is there a manuscript describing the method?**

The manuscript is being finalised. We hope to release the preprint describing the method in February 2025.
