"""
You will need to install java for TiDeTree.

Following https://github.com/seidels/tidetree : download it from https://www.azul.com/downloads/?package=jdk#zulu

Then, you can point the variable JAVA to your java installation.

You will also need to install beast, which can be done with conda with:
$ conda install beast -c bioconda
If this doesn't work, look at the BEAST installation guide.
"""
import os
from copy import deepcopy
import numpy as np
import networkx as nx
import subprocess
import tempfile
from cassiopeia.data import CassiopeiaTree
from cassiopeia.tools.branch_length_estimator import BranchLengthEstimator, IIDExponentialMLE
from typing import Dict, List, Optional
import pandas as pd
import wget
import contextlib
from casbench.branch_length_estimator._number_of_mutations_ble import NumberOfMutationsBLE
from casbench.ancestral_states_reconstructor._asr import maximum_parsimony

JAVA = "java"

def _get_xml_str(
    tree: CassiopeiaTree,
    priors: Dict[str, float],
    experiment_duration: float,
    edit_duration: float,
    chain_length: int,
    subsampling_probability: float = 1.0,
    silencing_rate: float = 0.0,
) -> str:
    """
    Build the xml file to use to run tidetree.

    Here `tree` is the initial tree topology (and branch lengths).
    """
    character_matrix = '\n'.join([
        f'<sequence id="{leaf_name}" spec="Sequence" taxon="{leaf_name}" value="' + f'{",".join([str(state) if state != -1 else "N" for state in list(tree.character_matrix.loc[leaf_name])])}' + '"/>'
        for leaf_name in sorted(tree.leaves)
    ])
    leaf_depths = ','.join([f"{leaf_name}={experiment_duration}" for leaf_name in sorted(tree.leaves)])
    newick_str = tree.get_newick(record_branch_lengths=True)
    number_of_states_including_missing = len(priors) + 1
    frequencies = "1" + ''.join([" 0"] * len(priors))  # I think these are the root frequencies, and this forces the root to be unmutated
    q_distribution = " ".join([str(priors[i]) for i in range(1, len(priors) + 1)])
    res = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<beast beautitemplate='MultiTypeBirthDeath' beautistatus=''
       namespace="beast.pkgmgmt
       :beast.base.core
       :beast.base.inference
       :beast.base.evolution.alignment
       :beast.base.evolution.tree.coalescent
       :beast.base.core
       :beast.base.inference.util
       :beast.evolution.nuc
       :beast.base.evolution.operator
       :beast.base.inference.operator
       :beast.base.evolution.sitemodel
       :beast.base.evolution.substitutionmodel
       :tidetree
       :beast.base.evolution.likelihood" required="" version="2.7">

<!--
This is an example xml that can be adapted to use TiDeTree on your data.
In this example, a population of cells was grown for 54h. Editing was induced from
time point 0 for 36h and 10 cells were sampled.

Running the analysis infers the tree linking the 10 cells and the cells' division and death rate.

Note that some words in this xml refer to time flowing either forward or backward.
Forward time is used to count the hours from the start of the experiment (so some time in the past)
toward the time when the cells were sampled. (e.g. in dateTrait, date-forward).
Backward time or height counts the hours the other way around, i.e. the duration between when the cells where sampled
and some time in the past. (e.g. the scarring height is the duration between sampling the cells and the time when
scarring started.
-->

<!-- syntactic sugar -->
<map name="Uniform" >beast.base.inference.distribution.Uniform</map>
<map name="Exponential" >beast.base.inference.distribution.Exponential</map>
<map name="LogNormal" >beast.base.inference.distribution.LogNormalDistributionModel</map>
<map name="Normal" >beast.base.inference.distribution.Normal</map>
<map name="Beta" >beast.base.inference.distribution.Beta</map>
<map name="Gamma" >beast.base.inference.distribution.Gamma</map>
<map name="LaplaceDistribution" >beast.base.inference.distribution.LaplaceDistribution</map>
<map name="prior" >beast.base.inference.distribution.Prior</map>
<map name="InverseGamma" >beast.base.inference.distribution.InverseGamma</map>
<map name="OneOnX" >beast.base.inference.distribution.OneOnX</map>

<!-- input data -->
<data  id="s1_c1_data.txt" spec="Alignment" name="alignment">
    <userDataType spec="tidetree.evolution.datatype.EditData"
                     nrOfStates="{number_of_states_including_missing}" />
{character_matrix}
</data>

<!-- define set of taxa based on alignment - cells in this context -->
<taxa id="TaxonSet" spec="TaxonSet">
    <alignment idref="s1_c1_data.txt"/>
</taxa>

<!-- specify tip dates, here 54h after the start of the experiment -->
<traitSet id="dateTrait" spec="beast.base.evolution.tree.TraitSet" taxa="@TaxonSet" traitname="date-forward"
	  value="{leaf_depths}"/>


  <substModel id="substModel"
               spec="tidetree.substitutionmodel.EditAndSilencingModel"
               editRates="@editRate" silencingRate="@silencingRate"
               editHeight="{experiment_duration}" editDuration="{edit_duration}">
    <frequencies spec="beast.base.evolution.substitutionmodel.Frequencies" frequencies="{frequencies}" estimate="false"/>
  </substModel>

  <siteModel spec="SiteModel" id="siteModel" mutationRate="@mutationRate" proportionInvariant="@proportionInvariant"
              gammaCategoryCount="0" substModel="@substModel">
  </siteModel>


<run id="mcmc" spec="MCMC" chainLength="{chain_length}">

    <!-- here the parameter of the model are defined that will be operated on (i.e. estimated) during the MCMC -->
    <state id="state" spec="State" storeEvery="5000">
    <!-- Fix the tree topology to GT -->
            <!-- <stateNode id="tree" spec="tidetree.tree.StartingTree" rootHeight="{experiment_duration}"
                       taxa="@s1_c1_data.txt" editHeight="{experiment_duration}"
                       editDuration="{edit_duration}" sequencesAreClustered="false"> -->
            <stateNode id='tree' spec='beast.base.evolution.tree.TreeParser' IsLabelledNewick='true' adjustTipHeights='false'
                    taxa='@s1_c1_data.txt'
                    newick="{newick_str}">
                <trait idref="dateTrait"/>
                <taxonset idref="TaxonSet"/>
            </stateNode>

        <!-- find the definition of the scarring, loss rate and clock rate in the TiDeTree manuscript -->
          <parameter id="editRate" spec="parameter.RealParameter"
		     dimension="1" lower="0.0" name="stateNode"
		     upper="1000"> {q_distribution}
          </parameter>
        <parameter id="silencingRate" spec="parameter.RealParameter" dimension="1"
                   lower="0.0" name="stateNode" upper="1000"> {silencing_rate}</parameter>

        <!-- the clock rate is the rate of acquiring edited state 2  -->
        <parameter id="clockRate.c" spec="parameter.RealParameter" name="stateNode">1.0</parameter>
        <!-- the mutation rate could be another rate multiplier, but we leave it fixed to 1 -->
        <parameter spec="parameter.RealParameter" estimate="false" id="mutationRate" name="stateNode">1.0</parameter>

        <parameter spec="parameter.RealParameter" estimate="false" lower="0.0" id="proportionInvariant"
                   name="stateNode" upper="1.0">0.0</parameter>

        <!-- the following parameters belong to the extended birth-death sampling model by Stadler2013 -->
        <!-- birth rate corresponds to cell division rate -->
        <parameter id="birthRate" spec="parameter.RealParameter" dimension="1" lower="0.0" name="stateNode" upper="5">0.6</parameter>
        <!-- death rate corresponds to apoptosis rate -->
        <parameter id="deathRate" spec="parameter.RealParameter" dimension="1" lower="0.0" name="stateNode" upper="5">0.01</parameter>
        <!-- sampling rate is a rate of sampling through time; since cells are only collected at single time point we set
        this to 0 and specify the sampling proportion at present: rho -->
        <parameter id="samplingRate" spec="parameter.RealParameter" lower="0.0" name="stateNode"
		   upper="1.0"> 0 </parameter>
        <parameter id="rho" spec="parameter.RealParameter" lower="0.0" name="stateNode" upper="1.0"> {subsampling_probability}</parameter>

        <!-- The origin is the start of the population process (see Stadler2013), i.e. in most applications the time that passed
         from the start of the experiment until the cells are sequenced. Here this is {experiment_duration}h. -->
        <parameter id="origin" spec="parameter.RealParameter" name="stateNode" upper="{experiment_duration + .0001}">{experiment_duration + .001}</parameter>

    </state>



    <distribution id="posterior" spec="beast.base.inference.CompoundDistribution">

        <!-- Prior distributions -->
        <distribution id="prior" spec="beast.base.inference.CompoundDistribution">

            <!-- The tree prior (or population dynamic model): the birth death sampling model -->
                <distribution id="birthDeathSamplingModel" spec="bdsky.evolution.speciation.BirthDeathSkylineModel"
                          conditionOnSurvival="True" tree="@tree" origin="@origin" contemp="true"
                          birthRate="@birthRate" deathRate="@deathRate" rho="@rho"
                          samplingRate="@samplingRate" >
                </distribution>

            <!-- prior distributions on the substitution model parameters -->
            <prior id="editRatePrior" name="distribution" x="@editRate">
                <LogNormal name="distr">
                  <parameter  spec="parameter.RealParameter" estimate="false" name="M">-4.0</parameter>
                  <parameter  spec="parameter.RealParameter" estimate="false" lower="0.0" name="S" upper="20.0">1.0</parameter>
                </LogNormal>
            </prior>

            <prior id="ClockPrior.c:LTv2_EBd32_cluster_1" name="distribution" x="@clockRate.c">
                <Uniform id="Uniform.0" name="distr" upper="Infinity"/>
            </prior>


            <!-- prior distributions on phylodynamic parameters -->
            <!-- [0.4, 2.7] -->
            <prior id="birthRatePrior" name="distribution" x="@birthRate">
                <LogNormal name="distr" M="0" S="0.6"/>
            </prior>
            <!-- [0.01, 1.4] -->
            <prior id="deathRatePrior" name="distribution" x="@deathRate">
               <LogNormal name="distr" M="-2" S="1.4"/>
            </prior>
        </distribution>

        <!-- Tree likelihood -->
        <distribution id="likelihood" spec="beast.base.inference.CompoundDistribution">

                <distribution id="treeLikelihood"
                              spec="tidetree.distributions.TreeLikelihoodWithEditWindow"
                              data="@s1_c1_data.txt" origin="@origin"
			      tree="@tree"
			      siteModel="@siteModel">
                    <branchRateModel spec="beast.base.evolution.branchratemodel.StrictClockModel" clock.rate="@clockRate.c"/>
                </distribution>

        </distribution>
    </distribution>

   <!-- tree operators -->
           <!-- operator  spec="tidetree.operators.WilsonBaldingSubTrees"
                      tree="@tree" weight="30.0" scarringStopHeight="18"/ -->

            <operator  spec="Uniform" tree="@tree" weight="30.0"/>

            <!-- <operator  spec="SubtreeSlide" tree="@tree" weight="3.0"/>
            <operator spec="SubtreeSlide" tree="@tree" weight="3.0" size="30"/>
            <operator  spec="Exchange" isNarrow="false" tree="@tree" weight="30.0"/> -->


	<!-- Operators on phylogenetic parameters-->
	<!-- operator on edit probabilities while keeping their sum fixed to 1 -->
	  <operator id="editRateScaler" spec="DeltaExchangeOperator" parameter="@editRate"
                 weight="3.0"/>
    <operator id="clockRateScaler" spec="ScaleOperator" parameter="@clockRate.c"
              scaleFactor="0.8" weight="3.0"/>


   <!-- Operators on phylodynamic parameters -->
    <operator id="birthRateScaler" spec="ScaleOperator" parameter="@birthRate" scaleFactor="0.8" weight="3.0"/>
    <operator id="deathRateScaler" spec="ScaleOperator" parameter="@deathRate" scaleFactor="0.8" weight="3.0"/>

    <operator id="updownBD.t:LTv2_EBd32_cluster_1" spec="UpDownOperator" scaleFactor="0.8" weight="30.0">
         <up idref="birthRate"/>
         <down idref="deathRate"/>
    </operator>


    <!-- Log parameters -->
    <logger id="tracelog" spec="Logger" fileName="$(filebase).$(seed).log" logEvery="10000">
        <log idref="posterior"/>
        <log idref="likelihood"/>
        <log idref="prior"/>
        <log idref="treeLikelihood"/>
        <log id="treeHeight"
                 spec="beast.base.evolution.tree.TreeHeightLogger"
                 tree="@tree"/>
        <log idref="editRate"/>
        <log idref="silencingRate" />
        <log idref="clockRate.c"/>
        <log idref="birthRate"/>
        <log idref="rho"/>
        <log idref="deathRate"/>
    </logger>

    <logger id="screenlog" spec="Logger" logEvery="1000">
        <log idref="posterior"/>
        <log id="ESS.0" spec="util.ESS" arg="@posterior"/>
        <log idref="likelihood"/>
        <log idref="prior"/>
    </logger>


    <!-- log tree -->
    <logger id="treelog.t:cluster" spec="Logger" fileName="$(filebase).$(tree).$(seed).trees" logEvery="10000" mode="tree">
                <log idref="tree" printMetaData="true"/>
    </logger>


</run>

</beast>

"""
    return res


@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


class TiDeTreeError(Exception):
    pass


def get_mean_newick_str(
    trees_file_contents: str,
) -> str:
    """
    Converts the tidetree output into the mean tree.
    """
    lines = trees_file_contents.split('\n')
    leaf_idx_to_leaf_name = {}
    mean_tree_times = None
    reference_tree = None
    num_trees = 0
    for i, line in enumerate(lines):
        if line.strip() == "Translate":
            j = i + 1
            while True:
                if lines[j].strip() == ";":
                    break
                else:
                    # This has a translation
                    leaf_idx, leaf_name = lines[j].strip().split(" ")
                    leaf_idx_to_leaf_name[leaf_idx] = leaf_name.strip(",")
                j += 1
        elif line.startswith("tree"):
            num_trees += 1
            tree_mcmc_sample = line.split(" ")[-1]
            # I need to do the renaming in two steps.
            # First I need to replace the integer names by the real names, making sure they are not integers so
            # that the relabeling doesn't clash.
            # So, if the translation is the following:
            # 0, 1, 2 -> 2, 1, 0
            # Then we do:
            # 0, 1, 2 -> 2-LEAF, 1-LEAF, 0-LEAF -> 2, 1, 0
            # which avoids clashes. Indeed, if we did not use this 2 step procedure, we would get:
            # 0, 1, 2 --(0 goes to 2)--> 2, 1, 2 --(1 goes to 1)-->2, 1, 2--(2 goes to 0)-->0, 1, 0 => translation failure does to naive sequential replacements!
            random_suffix = "LEAF"
            for leaf_idx, leaf_name in leaf_idx_to_leaf_name.items():
                if tree_mcmc_sample.count(f"({leaf_idx}:") + tree_mcmc_sample.count(f",{leaf_idx}:") != 1:
                    raise ValueError(
                        f'Checking for tree_mcmc_sample.count(f"({leaf_idx}:") + tree_mcmc_sample.count(f",{leaf_idx}:") == 1\n'
                        f'in {tree_mcmc_sample}'
                    )
                tree_mcmc_sample = tree_mcmc_sample.replace(f"({leaf_idx}:", f"({leaf_name}-{random_suffix}:")
                tree_mcmc_sample = tree_mcmc_sample.replace(f",{leaf_idx}:", f",{leaf_name}-{random_suffix}:")
            for leaf_idx, leaf_name in leaf_idx_to_leaf_name.items():
                if tree_mcmc_sample.count(f"({leaf_name}-{random_suffix}:") + tree_mcmc_sample.count(f",{leaf_name}-{random_suffix}:") != 1:
                    raise ValueError(
                        f'Checking for tree_mcmc_sample.count(f"({leaf_name}-{random_suffix}:") + tree_mcmc_sample.count(f",{leaf_name}-{random_suffix}:") == 1\n'
                        f'in {tree_mcmc_sample}'
                    )
                tree_mcmc_sample = tree_mcmc_sample.replace(f"({leaf_name}-{random_suffix}:", f"({leaf_name}:")
                tree_mcmc_sample = tree_mcmc_sample.replace(f",{leaf_name}-{random_suffix}:", f",{leaf_name}:")
            cassiopeia_tree = CassiopeiaTree(tree=tree_mcmc_sample)
            tree_mcmc_sample_times = cassiopeia_tree.get_times()
            if mean_tree_times is None:
                mean_tree_times = tree_mcmc_sample_times
                reference_tree = cassiopeia_tree
            else:
                # Need to add the new times
                # Make sure the edges are all the same though!
                assert(sorted(reference_tree.nodes) == sorted(cassiopeia_tree.nodes))
                for node in reference_tree.nodes:
                    if node != reference_tree.root:
                        assert(
                            reference_tree.parent(node) == cassiopeia_tree.parent(node)
                        )
                assert(sorted(mean_tree_times.keys()) == sorted(tree_mcmc_sample_times.keys()))
                for node_name, node_time in tree_mcmc_sample_times.items():
                    mean_tree_times[node_name] += node_time
    for node_name, node_time in list(mean_tree_times.items()):
        mean_tree_times[node_name] = node_time / num_trees
    reference_tree.set_times(mean_tree_times)
    res = reference_tree.get_newick(record_branch_lengths=True)
    return res


def test_get_mean_newick_str():
    res = get_mean_newick_str(
"""
#NEXUS

Begin taxa;
	Dimensions ntax=9;
		Taxlabels
			10_0112212221
			1_2012212021
			2_2112212021
			3_2112212021
			4_2112212021
			5_0012212221
			6_0012012221
			8_2120010021
			9_2120010021
			;
End;
Begin trees;
	Translate
		   1 10_0112212221,
		   2 1_2012212021,
		   3 2_2112212021,
		   4 3_2112212021,
		   5 4_2112212021,
		   6 5_0012212221,
		   7 6_0012012221,
		   8 8_2120010021,
		   9 9_2120010021
;
tree STATE_0 = ((1:0.5441176470588236,(8:0.16911764705882348,9:0.16911764705882348):0.3750000000000001):0.4558823529411764,(((2:0.07352941176470584,3:0.07352941176470584):0.27941176470588236,(4:0.044117647058823484,5:0.044117647058823484):0.3088235294117647):0.3088235294117647,(6:0.022058823529411797,7:0.022058823529411797):0.6397058823529411):0.3382352941176471):0.0;
tree STATE_10000 = ((1:0.8033945857287271,(8:0.23136644538463222,9:0.23136644538463222):0.5720281403440949):0.19660541427127287,(((2:0.17524259135176398,3:0.17524259135176398):0.134635852686776,(4:0.1552434055653545,5:0.1552434055653545):0.1546350384731855):0.3444464561279071,(6:0.46590317779119195,7:0.46590317779119195):0.18842172237525512):0.34567509983355293):0.0;
tree STATE_20000 = ((1:0.7267249542251888,(8:0.15266207016514746,9:0.15266207016514746):0.5740628840600414):0.2732750457748112,(((2:0.1022098045141354,3:0.1022098045141354):0.48637558601604836,(4:0.13566358752952534,5:0.13566358752952534):0.45292180300065843):0.16518405701379402,(6:0.3117838057924493,7:0.3117838057924493):0.4419856417515285):0.2462305524560222):0.0;
tree STATE_30000 = ((1:0.9970551974883924,(8:0.06715245258984183,9:0.06715245258984183):0.9299027448985506):0.0029448025116075582,(((2:0.18389013676691401,3:0.18389013676691401):0.27175257819260995,(4:0.2525873675825349,5:0.2525873675825349):0.20305534737698905):0.29656873381551047,(6:0.1652291147264873,7:0.1652291147264873):0.5869823340485472):0.24778855122496557):0.0;
tree STATE_40000 = ((1:0.8617861002775755,(8:0.08775781350462265,9:0.08775781350462265):0.7740282867729529):0.13821389972242448,(((2:0.33200302683120714,3:0.33200302683120714):0.2087576962349595,(4:0.21286916476346437,5:0.21286916476346437):0.3278915583027022):0.34434265143868426,(6:0.7585540254805817,7:0.7585540254805817):0.12654934902426918):0.11489662549514912):0.0;
tree STATE_50000 = ((1:0.958397296399642,(8:0.4308700858396262,9:0.4308700858396262):0.5275272105600157):0.041602703600358004,(((2:0.4389792392729589,3:0.4389792392729589):0.035163325390914046,(4:0.20510971695790428,5:0.20510971695790428):0.2690328477059687):0.2032393503394757,(6:0.09509597183777564,7:0.09509597183777564):0.582285943165573):0.32261808499665134):0.0;
tree STATE_60000 = ((1:0.695343804927943,(8:0.07094476495855012,9:0.07094476495855012):0.6243990399693928):0.304656195072057,(((2:0.12938798324515213,3:0.12938798324515213):0.3430086363985698,(4:0.05024866778195114,5:0.05024866778195114):0.4221479518617708):0.1731181475552248,(6:0.19862431663133856,7:0.19862431663133856):0.4468904505676082):0.35448523280105326):0.0;
tree STATE_70000 = ((1:0.7397854164530702,(8:0.023720735689832052,9:0.023720735689832052):0.7160646807632381):0.26021458354692983,(((2:0.409275603576191,3:0.409275603576191):0.10490638215270692,(4:0.03847202190906865,5:0.03847202190906865):0.47570996381982933):0.05532928714076224,(6:0.045139555640043016,7:0.045139555640043016):0.5243717172296172):0.4304887271303398):0.0;
tree STATE_80000 = ((1:0.9504365413348564,(8:0.07374047152366621,9:0.07374047152366621):0.8766960698111902):0.049563458665143556,(((2:0.13023919489839691,3:0.13023919489839691):0.05093084852782093,(4:0.04730865119462724,5:0.04730865119462724):0.1338613922315906):0.5765486600824241,(6:0.2715060285514297,7:0.2715060285514297):0.4862126749572122):0.2422812964913581):0.0;
tree STATE_90000 = ((1:0.9129036811629156,(8:0.09598542715487479,9:0.09598542715487479):0.8169182540080409):0.08709631883708435,(((2:0.49121018278167417,3:0.49121018278167417):0.035482270133644356,(4:0.10804706880508054,5:0.10804706880508054):0.418645384110238):0.17267629758964254,(6:0.4711818154239739,7:0.4711818154239739):0.22818693508098714):0.30063124949503894):0.0;
tree STATE_100000 = ((1:0.8308403816722437,(8:0.11838746980083928,9:0.11838746980083928):0.7124529118714045):0.16915961832775628,(((2:0.14945941823176187,3:0.14945941823176187):0.39858817887230225,(4:0.276771396156061,5:0.276771396156061):0.27127620094800314):0.11503471983667701,(6:0.6035168906892154,7:0.6035168906892154):0.05956542625152572):0.33691768305925884):0.0;
tree STATE_110000 = ((1:0.6889793584641323,(8:0.17463144539964082,9:0.17463144539964082):0.5143479130644915):0.3110206415358677,(((2:0.36451569259241245,3:0.36451569259241245):0.1818010410694787,(4:0.15291589790727858,5:0.15291589790727858):0.3934008357546126):0.3216767803549213,(6:0.22592628739425588,7:0.22592628739425588):0.6420672266225566):0.13200648598318754):0.0;
tree STATE_120000 = ((1:0.8981748763008797,(8:0.17833036740030225,9:0.17833036740030225):0.7198445089005774):0.10182512369912033,(((2:0.10339931063756055,3:0.10339931063756055):0.07596152662831088,(4:0.011435464394371377,5:0.011435464394371377):0.16792537287150006):0.4507890813209365,(6:0.2671928674047552,7:0.2671928674047552):0.36295705118205274):0.3698500814131921):0.0;
tree STATE_130000 = ((1:0.7538516069906096,(8:0.13132303910863982,9:0.13132303910863982):0.6225285678819698):0.2461483930093904,(((2:0.07656198153070584,3:0.07656198153070584):0.07812609746217891,(4:0.12678683925386813,5:0.12678683925386813):0.02790123973901662):0.5280815679887171,(6:0.2231045322087319,7:0.2231045322087319):0.4596651147728699):0.3172303530183982):0.0;
tree STATE_140000 = ((1:0.8344990767617456,(8:0.12535834152975492,9:0.12535834152975492):0.7091407352319907):0.16550092323825438,(((2:0.5081455042606509,3:0.5081455042606509):0.060903943474315714,(4:3.713537095504268E-4,5:3.713537095504268E-4):0.5686780940254161):0.2387107056886082,(6:0.6441040080219321,7:0.6441040080219321):0.16365614540164264):0.19223984657642523):0.0;
tree STATE_150000 = ((1:0.6733356290343171,(8:0.1634419464566511,9:0.1634419464566511):0.509893682577666):0.3266643709656829,(((2:0.28389195676499146,3:0.28389195676499146):0.2165846341437827,(4:0.11914142925560141,5:0.11914142925560141):0.38133516165317277):0.22974957587987144,(6:0.25643489414322856,7:0.25643489414322856):0.47379127264541704):0.2697738332113544):0.0;
tree STATE_160000 = ((1:0.9218930188346198,(8:0.06609267722265633,9:0.06609267722265633):0.8558003416119635):0.07810698116538017,(((2:0.12841728740179528,3:0.12841728740179528):0.16277330946546595,(4:0.13155752112984173,5:0.13155752112984173):0.1596330757374195):0.5340814117571739,(6:0.21470733683412888,7:0.21470733683412888):0.6105646717903063):0.17472799137556483):0.0;
tree STATE_170000 = ((1:0.7936329608914069,(8:0.04857532248245205,9:0.04857532248245205):0.7450576384089549):0.2063670391085931,(((2:0.18844652180558621,3:0.18844652180558621):0.1149467164515118,(4:0.19557203997306258,5:0.19557203997306258):0.10782119828403544):0.4691323078584026,(6:0.47805593435371596,7:0.47805593435371596):0.29446961176178466):0.22747445388449938):0.0;
tree STATE_180000 = ((1:0.9278451935870538,(8:0.025837305341905828,9:0.025837305341905828):0.9020078882451479):0.07215480641294625,(((2:0.340807965610104,3:0.340807965610104):0.01254855040233166,(4:0.337027618333311,5:0.337027618333311):0.016328897679124654):0.28846470509513367,(6:0.26830392674474873,7:0.26830392674474873):0.3735172943628206):0.35817877889243066):0.0;
tree STATE_190000 = ((1:0.9506353628579782,(8:0.04383229008325744,9:0.04383229008325744):0.9068030727747208):0.04936463714202177,(((2:0.4561539473065811,3:0.4561539473065811):0.0455490973986834,(4:0.4645612209889804,5:0.4645612209889804):0.037141823716284106):0.24858855363245635,(6:0.3460221763246543,7:0.3460221763246543):0.40426942201306654):0.24970840166227914):0.0;
tree STATE_200000 = ((1:0.8430685461508527,(8:0.15230127235860208,9:0.15230127235860208):0.6907672737922506):0.1569314538491473,(((2:0.053256605610291496,3:0.053256605610291496):0.3221721039516694,(4:0.06328462651764284,5:0.06328462651764284):0.31214408304431807):0.2736668587703574,(6:0.34227439159615697,7:0.34227439159615697):0.3068211767361613):0.3509044316676817):0.0;
tree STATE_210000 = ((1:0.9418738344911785,(8:0.20667303176111146,9:0.20667303176111146):0.7352008027300669):0.05812616550882155,(((2:0.2067021663240676,3:0.2067021663240676):0.0929898671622447,(4:0.06850719407575391,5:0.06850719407575391):0.2311848394105584):0.428819844759361,(6:0.19275131764966189,7:0.19275131764966189):0.5357605605960114):0.2714881217543267):0.0;
tree STATE_220000 = ((1:0.7704524018555464,(8:0.22156650002632194,9:0.22156650002632194):0.5488859018292245):0.22954759814445358,(((2:0.24828519145723335,3:0.24828519145723335):0.10913707502862727,(4:0.19882858075722462,5:0.19882858075722462):0.158593685728636):0.299157782905763,(6:0.24188155049287083,7:0.24188155049287083):0.41469849889875277):0.3434199506083764):0.0;
tree STATE_230000 = ((1:0.7804714624174149,(8:0.24783720823432392,9:0.24783720823432392):0.5326342541830911):0.21952853758258506,(((2:0.2839797818364504,3:0.2839797818364504):0.15666504132161002,(4:0.008916136730386867,5:0.008916136730386867):0.43172868642767354):0.20239126934528573,(6:0.3581093165529111,7:0.3581093165529111):0.28492677595043503):0.35696390749665385):0.0;
tree STATE_240000 = ((1:0.8799642152181208,(8:0.11925812982993764,9:0.11925812982993764):0.7607060853881832):0.12003578478187915,(((2:0.12054478330111265,3:0.12054478330111265):0.037501692994016095,(4:0.08406046123544106,5:0.08406046123544106):0.07398601505968769):0.38550100188025976,(6:0.21493758453429396,7:0.21493758453429396):0.32860989364109455):0.4564525218246115):0.0;
tree STATE_250000 = ((1:0.8425717912835737,(8:0.016186507877055844,9:0.016186507877055844):0.8263852834065178):0.15742820871642627,(((2:0.2936916051771768,3:0.2936916051771768):0.13893213377245833,(4:0.023117113318326537,5:0.023117113318326537):0.4095066256313086):0.2272346017915411,(6:0.33778639612621103,7:0.33778639612621103):0.3220719446149652):0.3401416592588238):0.0;
tree STATE_260000 = ((1:0.6571577356879552,(8:0.10964287361821234,9:0.10964287361821234):0.5475148620697429):0.3428422643120448,(((2:0.1767119895944854,3:0.1767119895944854):0.11468900949024433,(4:0.0732092566894786,5:0.0732092566894786):0.21819174239525113):0.4341947040826478,(6:0.1908954047997427,7:0.1908954047997427):0.5347002983676348):0.27440429683262246):0.0;
tree STATE_270000 = ((1:0.6054342770957644,(8:0.17255182970429706,9:0.17255182970429706):0.4328824473914673):0.39456572290423564,(((2:0.30653873704077295,3:0.30653873704077295):0.00971257149325322,(4:0.07042010323061167,5:0.07042010323061167):0.2458312053034145):0.19149610950239426,(6:0.07548398316794663,7:0.07548398316794663):0.4322634348684738):0.49225258196357957):0.0;
tree STATE_280000 = ((1:0.9120735901230899,(8:0.13552594786036187,9:0.13552594786036187):0.776547642262728):0.08792640987691014,(((2:0.20483233785061464,3:0.20483233785061464):0.018862421283417835,(4:0.040634154627506254,5:0.040634154627506254):0.18306060450652623):0.07228598871519332,(6:0.09641854403295697,7:0.09641854403295697):0.19956220381626882):0.7040192521507742):0.0;
tree STATE_290000 = ((1:0.674478992349523,(8:0.01020663658205573,9:0.01020663658205573):0.6642723557674672):0.325521007650477,(((2:0.08169532066695454,3:0.08169532066695454):0.24888636838039638,(4:0.21235636704241143,5:0.21235636704241143):0.11822532200493949):0.20061316361865017,(6:0.1486218130273694,7:0.1486218130273694):0.38257303963863165):0.4688051473339989):0.0;
tree STATE_300000 = ((1:0.7689995844193607,(8:0.19297082158782303,9:0.19297082158782303):0.5760287628315377):0.23100041558063933,(((2:0.20781352706877496,3:0.20781352706877496):0.1627317745844228,(4:0.2286625907647968,5:0.2286625907647968):0.14188271088840096):0.3537625466655243,(6:0.5698137659184892,7:0.5698137659184892):0.15449408240023288):0.2756921516812779):0.0;
tree STATE_310000 = ((1:0.938978869876254,(8:0.059723312928846015,9:0.059723312928846015):0.879255556947408):0.06102113012374599,(((2:0.10741096601074586,3:0.10741096601074586):0.1214037467103314,(4:0.0010541054065220919,5:0.0010541054065220919):0.22776060731455516):0.2684631418999127,(6:0.0673236666403687,7:0.0673236666403687):0.42995418798062124):0.50272214537901):0.0;
tree STATE_320000 = ((1:0.7196168086118109,(8:0.0100650955081459,9:0.0100650955081459):0.7095517131036649):0.2803831913881891,(((2:0.17309073482907822,3:0.17309073482907822):0.09463219617075347,(4:0.04902753254436032,5:0.04902753254436032):0.21869539845547137):0.5194227776010548,(6:0.6343782025501092,7:0.6343782025501092):0.15276750605077727):0.2128542913991135):0.0;
tree STATE_330000 = ((1:0.7474596529700008,(8:0.05517972273287139,9:0.05517972273287139):0.6922799302371294):0.25254034702999917,(((2:0.20526524380051905,3:0.20526524380051905):0.029072100144673457,(4:0.008563091787465588,5:0.008563091787465588):0.22577425215772692):0.31877134993336503,(6:0.1507252412110666,7:0.1507252412110666):0.402383452667491):0.44689130612144246):0.0;
tree STATE_340000 = ((1:0.9509342520958685,(8:0.03647964500863491,9:0.03647964500863491):0.9144546070872336):0.04906574790413154,(((2:0.302806910757973,3:0.302806910757973):0.04748996016072338,(4:0.0031158115120783784,5:0.0031158115120783784):0.347181059406618):0.41802904786878475,(6:0.23940196547522366,7:0.23940196547522366):0.5289239533122575):0.23167408121251887):0.0;
tree STATE_350000 = ((1:0.7211063445917002,(8:0.1765199378847383,9:0.1765199378847383):0.5445864067069619):0.2788936554082998,(((2:0.17384047981222547,3:0.17384047981222547):0.16085699886252816,(4:0.03634153431295785,5:0.03634153431295785):0.2983559443617958):0.3336458386585073,(6:0.32271365772043714,7:0.32271365772043714):0.34562965961282377):0.3316566826667391):0.0;
tree STATE_360000 = ((1:0.6024720353829581,(8:0.17740358290959,9:0.17740358290959):0.4250684524733681):0.3975279646170419,(((2:0.1869950035092703,3:0.1869950035092703):0.30172890685833076,(4:0.21450758177071508,5:0.21450758177071508):0.27421632859688605):0.2806379553407329,(6:0.3130645253703894,7:0.3130645253703894):0.45629734033794456):0.23063813429166602):0.0;
tree STATE_370000 = ((1:0.9132806163699774,(8:0.2289674668707394,9:0.2289674668707394):0.6843131494992379):0.08671938363002263,(((2:0.39826254001349737,3:0.39826254001349737):0.04262770931963239,(4:0.05122831361017841,5:0.05122831361017841):0.3896619357229513):0.34982864590384083,(6:0.12523272960860504,7:0.12523272960860504):0.6654861656283655):0.2092811047630294):0.0;
tree STATE_380000 = ((1:0.8912583699691554,(8:0.283435753828853,9:0.283435753828853):0.6078226161403024):0.10874163003084458,(((2:0.0917070650799501,3:0.0917070650799501):0.027423922796996833,(4:0.047920759512407096,5:0.047920759512407096):0.07121022836453984):0.40653779788837724,(6:0.14638121566570747,7:0.14638121566570747):0.37928757009961667):0.47433121423467584):0.0;
tree STATE_390000 = ((1:0.8487797563836645,(8:0.37825411365942535,9:0.37825411365942535):0.4705256427242392):0.15122024361633546,(((2:0.17882205105603163,3:0.17882205105603163):0.2510517084956253,(4:0.02100238355611841,5:0.02100238355611841):0.4088713759955386):0.16163294051686772,(6:0.31727482235980864,7:0.31727482235980864):0.27423187770871604):0.4084932999314753):0.0;
tree STATE_400000 = ((1:0.810403182467512,(8:0.18299677433918868,9:0.18299677433918868):0.6274064081283233):0.18959681753248803,(((2:0.07986646298422662,3:0.07986646298422662):0.38418554898446583,(4:0.2348635593608835,5:0.2348635593608835):0.22918845260780896):0.2063173966628727,(6:0.30905864537742056,7:0.30905864537742056):0.3613107632541446):0.32963059136843487):0.0;
tree STATE_410000 = ((1:0.9944346374032226,(8:0.11221009327041502,9:0.11221009327041502):0.8822245441328076):0.005565362596777423,(((2:0.20953852613246235,3:0.20953852613246235):0.023187570547278796,(4:0.0836780850127104,5:0.0836780850127104):0.14904801166703074):0.40985991251445436,(6:0.19840138592417475,7:0.19840138592417475):0.44418462327002073):0.3574139908058045):0.0;
tree STATE_420000 = ((1:0.9065324783069283,(8:0.2673754526748098,9:0.2673754526748098):0.6391570256321185):0.09346752169307171,(((2:0.11735551798507272,3:0.11735551798507272):0.11120591104238346,(4:0.02873134279246354,5:0.02873134279246354):0.19983008623499265):0.13828627548275557,(6:0.30286771397545903,7:0.30286771397545903):0.06397999053475273):0.6331522954897882):0.0;
tree STATE_430000 = ((1:0.9450236391879202,(8:0.2027734789016005,9:0.2027734789016005):0.7422501602863197):0.054976360812079794,(((2:0.29960351304505156,3:0.29960351304505156):0.017371073178332253,(4:0.12324485354623729,5:0.12324485354623729):0.19372973267714652):0.4636816234486828,(6:0.4261057652809277,7:0.4261057652809277):0.3545504443911389):0.21934379032793339):0.0;
tree STATE_440000 = ((1:0.9138249631506612,(8:0.0876317723248936,9:0.0876317723248936):0.8261931908257676):0.0861750368493388,(((2:0.4252851633829661,3:0.4252851633829661):0.03222699385527489,(4:0.03517129621656044,5:0.03517129621656044):0.42234086102168056):0.3185969973995796,(6:0.14054595506187612,7:0.14054595506187612):0.6355631995759445):0.2238908453621794):0.0;
tree STATE_450000 = ((1:0.7253213196447559,(8:0.026649359617211855,9:0.026649359617211855):0.6986719600275441):0.2746786803552441,(((2:0.10967506709976735,3:0.10967506709976735):0.22982787026619686,(4:0.11922627856138286,5:0.11922627856138286):0.22027665880458136):0.22600649569334624,(6:0.43918399862765733,7:0.43918399862765733):0.12632543443165312):0.43449056694068955):0.0;
tree STATE_460000 = ((1:0.7767606637553297,(8:0.05525681815229635,9:0.05525681815229635):0.7215038456030334):0.2232393362446703,(((2:0.3000280290774541,3:0.3000280290774541):0.3435596390069725,(4:0.04283695503995258,5:0.04283695503995258):0.600750713044474):0.07640982472808844,(6:0.17633451321988086,7:0.17633451321988086):0.5436629795926342):0.280002507187485):0.0;
tree STATE_470000 = ((1:0.8084748194633146,(8:0.1304832107193302,9:0.1304832107193302):0.6779916087439843):0.19152518053668544,(((2:0.17538865039476514,3:0.17538865039476514):0.06301602391767258,(4:0.011561963401623186,5:0.011561963401623186):0.22684271091081454):0.25256010355175196,(6:0.08870351633618384,7:0.08870351633618384):0.40226126152800584):0.5090352221358103):0.0;
tree STATE_480000 = ((1:0.7200259915500877,(8:0.19819462596197163,9:0.19819462596197163):0.5218313655881162):0.27997400844991227,(((2:0.13013978868104803,3:0.13013978868104803):0.3203030731259505,(4:0.17156831961370891,5:0.17156831961370891):0.27887454219328967):0.0858913417263315,(6:0.30133164863418693,7:0.30133164863418693):0.23500255489914312):0.46366579646666994):0.0;
tree STATE_490000 = ((1:0.540302148610025,(8:0.04466340968109882,9:0.04466340968109882):0.4956387389289262):0.459697851389975,(((2:0.139696621689969,3:0.139696621689969):0.1815976209079232,(4:0.2037540624217639,5:0.2037540624217639):0.1175401801761283):0.4652053891688839,(6:0.2325801993299366,7:0.2325801993299366):0.5539194324368395):0.21350036823322394):0.0;
tree STATE_500000 = ((1:0.8518086484908528,(8:0.01055432578848278,9:0.01055432578848278):0.84125432270237):0.1481913515091472,(((2:0.16728004236934935,3:0.16728004236934935):0.03708376240768976,(4:0.14549041042595978,5:0.14549041042595978):0.05887339435107933):0.4250403121893065,(6:0.37894563327124237,7:0.37894563327124237):0.2504584836951032):0.3705958830336544):0.0;
tree STATE_510000 = ((1:0.5946235586794293,(8:0.08025336654776764,9:0.08025336654776764):0.5143701921316617):0.4053764413205707,(((2:0.06602670469059209,3:0.06602670469059209):0.06632375942026168,(4:9.104367747365237E-4,5:9.104367747365237E-4):0.13144002733611726):0.33649231061124313,(6:0.3812401780103967,7:0.3812401780103967):0.08760259671170018):0.5311572252779031):0.0;
tree STATE_520000 = ((1:0.9133144139940073,(8:0.03625788271793784,9:0.03625788271793784):0.8770565312760694):0.08668558600599274,(((2:0.10885405643752895,3:0.10885405643752895):0.24953587248642267,(4:0.09152290886196843,5:0.09152290886196843):0.26686702006198315):0.42190969407074475,(6:0.47923049711535404,7:0.47923049711535404):0.3010691258793423):0.21970037700530365):0.0;
tree STATE_530000 = ((1:0.9529334057498413,(8:0.3034645784089779,9:0.3034645784089779):0.6494688273408634):0.04706659425015869,(((2:0.1720205135581721,3:0.1720205135581721):0.014039749582022643,(4:0.013957988078501527,5:0.013957988078501527):0.17210227506169323):0.3975290028951487,(6:0.09703557961103776,7:0.09703557961103776):0.4865536864243057):0.41641073396465655):0.0;
tree STATE_540000 = ((1:0.7972890058900132,(8:0.10640380142179443,9:0.10640380142179443):0.6908852044682188):0.20271099410998683,(((2:0.24151679661294329,3:0.24151679661294329):0.01842861034611254,(4:0.23261453235661622,5:0.23261453235661622):0.027330874602439603):0.34406274325852765,(6:0.25792220784736813,7:0.25792220784736813):0.34608594237021534):0.3959918497824165):0.0;
tree STATE_550000 = ((1:0.9077019260246075,(8:0.14099999195885896,9:0.14099999195885896):0.7667019340657486):0.09229807397539247,(((2:0.11408963357741685,3:0.11408963357741685):0.18245449422836857,(4:0.11265616963559524,5:0.11265616963559524):0.18388795817019019):0.33602373549023046,(6:0.2937727566771627,7:0.2937727566771627):0.3387951066188532):0.3674321367039841):0.0;
tree STATE_560000 = ((1:0.7644999755894559,(8:0.03280535883203576,9:0.03280535883203576):0.7316946167574202):0.23550002441054407,(((2:0.4677099016752933,3:0.4677099016752933):0.006568712783696085,(4:0.019679458181999576,5:0.019679458181999576):0.4545991562769898):0.2611032361109474,(6:0.34120418723722573,7:0.34120418723722573):0.39417766333271104):0.26461814943006323):0.0;
tree STATE_570000 = ((1:0.8855401489465905,(8:7.529621810152892E-4,9:7.529621810152892E-4):0.8847871867655752):0.1144598510534095,(((2:0.2445542254494322,3:0.2445542254494322):0.2375560276135215,(4:0.03862550808995298,5:0.03862550808995298):0.4434847449730007):0.3299224875111128,(6:0.41361702055857547,7:0.41361702055857547):0.398415720015491):0.1879672594259335):0.0;
tree STATE_580000 = ((1:0.9231475842952583,(8:0.15017892122242915,9:0.15017892122242915):0.7729686630728292):0.07685241570474166,(((2:0.07367825594487282,3:0.07367825594487282):0.24235392021342872,(4:0.006634639114058528,5:0.006634639114058528):0.309397537044243):0.5693202246394302,(6:0.2008449186269874,7:0.2008449186269874):0.6845074821707443):0.1146475992022683):0.0;
tree STATE_590000 = ((1:0.6457790169088515,(8:0.238652933268326,9:0.238652933268326):0.4071260836405255):0.35422098309114847,(((2:0.28944958241744934,3:0.28944958241744934):0.21959144675746922,(4:0.046501558816548876,5:0.046501558816548876):0.4625394703583697):0.10124435862119197,(6:0.12346906419759633,7:0.12346906419759633):0.4868163235985142):0.3897146122038895):0.0;
tree STATE_600000 = ((1:0.9316652400030346,(8:0.05745685477991192,9:0.05745685477991192):0.8742083852231227):0.06833475999696537,(((2:0.36148959051813745,3:0.36148959051813745):0.16631098798840482,(4:0.2929427857516241,5:0.2929427857516241):0.2348577927549182):0.13752552578084554,(6:0.5597540767470387,7:0.5597540767470387):0.10557202754034911):0.3346738957126122):0.0;
tree STATE_610000 = ((1:0.7912324914242403,(8:0.024201549458793437,9:0.024201549458793437):0.7670309419654469):0.20876750857575965,(((2:0.1484953688043541,3:0.1484953688043541):0.1652518636606345,(4:0.07267503916499427,5:0.07267503916499427):0.24107219329999433):0.28922975178813326,(6:0.16722717755139332,7:0.16722717755139332):0.4357498067017285):0.39702301574687815):0.0;
tree STATE_620000 = ((1:0.9157062542197727,(8:0.5385406175506593,9:0.5385406175506593):0.3771656366691134):0.0842937457802273,(((2:0.2969308642490256,3:0.2969308642490256):0.16291883088058845,(4:0.007572135332650689,5:0.007572135332650689):0.4522775597969634):0.45666874782973177,(6:0.22635710929544078,7:0.22635710929544078):0.690161333663905):0.08348155704065419):0.0;
tree STATE_630000 = ((1:0.9161979740583353,(8:0.33681439239397604,9:0.33681439239397604):0.5793835816643592):0.08380202594166475,(((2:0.09521483330101599,3:0.09521483330101599):0.2523874128965927,(4:0.10170593539095485,5:0.10170593539095485):0.24589631080665386):0.19347803055945567,(6:0.45532870537361997,7:0.45532870537361997):0.08575157138344441):0.4589197232429356):0.0;
tree STATE_640000 = ((1:0.8501122876784978,(8:0.02125917798883283,9:0.02125917798883283):0.828853109689665):0.14988771232150222,(((2:0.17941510463767169,3:0.17941510463767169):0.08323199646602136,(4:0.008726469308198368,5:0.008726469308198368):0.2539206317954947):0.15989888441983735,(6:0.2188262226105086,7:0.2188262226105086):0.2037197629130218):0.5774540144764696):0.0;
tree STATE_650000 = ((1:0.8780198758028144,(8:0.1417579127021721,9:0.1417579127021721):0.7362619631006423):0.12198012419718562,(((2:0.22323928884456393,3:0.22323928884456393):0.2666825383687931,(4:0.1813110753676622,5:0.1813110753676622):0.30861075184569486):0.22288919776493704,(6:0.31846697359554577,7:0.31846697359554577):0.3943440513827483):0.28718897502170593):0.0;
tree STATE_660000 = ((1:0.9532427878920826,(8:0.033955754889911184,9:0.033955754889911184):0.9192870330021714):0.04675721210791739,(((2:0.24858479182742121,3:0.24858479182742121):0.15100977970541568,(4:0.01463177306672704,5:0.01463177306672704):0.38496279846610987):0.25103407172987285,(6:0.35445680205235125,7:0.35445680205235125):0.2961718412103585):0.34937135673729025):0.0;
tree STATE_670000 = ((1:0.8571461925991426,(8:0.11464861944201107,9:0.11464861944201107):0.7424975731571315):0.1428538074008574,(((2:0.1158107218631801,3:0.1158107218631801):0.08128689193517656,(4:0.020592834592417023,5:0.020592834592417023):0.17650477920593965):0.44627763945554366,(6:0.2172905509892045,7:0.2172905509892045):0.42608470226469586):0.3566247467460997):0.0;
tree STATE_680000 = ((1:0.6088862532383883,(8:0.4486269644044001,9:0.4486269644044001):0.1602592888339882):0.3911137467616117,(((2:0.02564654286985019,3:0.02564654286985019):0.17463132895063038,(4:0.01083141909071262,5:0.01083141909071262):0.18944645272976796):0.5704764124193424,(6:0.3774487040684469,7:0.3774487040684469):0.39330558017137607):0.22924571576017705):0.0;
tree STATE_690000 = ((1:0.8524744350538693,(8:0.14523893901651397,9:0.14523893901651397):0.7072354960373554):0.14752556494613067,(((2:0.32723949695442467,3:0.32723949695442467):0.008086601229199242,(4:0.012899315720113894,5:0.012899315720113894):0.32242678246351003):0.2399370819998473,(6:0.20460240937593677,7:0.20460240937593677):0.37066077080753446):0.4247368198165288):0.0;
tree STATE_700000 = ((1:0.6316537404475896,(8:0.009536586573195128,9:0.009536586573195128):0.6221171538743945):0.3683462595524104,(((2:0.26554736810637175,3:0.26554736810637175):0.13594408159619764,(4:0.08691936183453357,5:0.08691936183453357):0.3145720878680358):0.22232434882462543,(6:0.34878405320333156,7:0.34878405320333156):0.27503174532386326):0.3761842014728052):0.0;
tree STATE_710000 = ((1:0.9394878546917029,(8:0.8327412486313344,9:0.8327412486313344):0.10674660606036857):0.06051214530829707,(((2:0.18175746010553986,3:0.18175746010553986):0.16274416264962296,(4:0.13055233608174796,5:0.13055233608174796):0.21394928667341487):0.37888579908816256,(6:0.2600980732750095,7:0.2600980732750095):0.4632893485683159):0.2766125781566746):0.0;
tree STATE_720000 = ((1:0.8280318190569733,(8:0.0035943731236078524,9:0.0035943731236078524):0.8244374459333654):0.17196818094302668,(((2:0.22847801752804636,3:0.22847801752804636):0.02814284372643866,(4:0.07809604256961766,5:0.07809604256961766):0.17852481868486736):0.31141608998295045,(6:0.2831262015117758,7:0.2831262015117758):0.28491074972565966):0.4319630487625645):0.0;
tree STATE_730000 = ((1:0.7497254616388888,(8:0.023503336046357872,9:0.023503336046357872):0.7262221255925309):0.25027453836111124,(((2:0.03503687184880045,3:0.03503687184880045):0.1576242175779346,(4:0.09865110541519374,5:0.09865110541519374):0.0940099840115413):0.34991985322505564,(6:0.3222296860471273,7:0.3222296860471273):0.22035125660466337):0.4574190573482093):0.0;
tree STATE_740000 = ((1:0.9171601372396766,(8:0.3167594416319299,9:0.3167594416319299):0.6004006956077467):0.0828398627603234,(((2:0.2655188439938904,3:0.2655188439938904):0.11502228291430155,(4:0.017906919983612633,5:0.017906919983612633):0.3626342069245793):0.23597201995073136,(6:0.08521712897592407,7:0.08521712897592407):0.5312960178829993):0.38348685314107667):0.0;
tree STATE_750000 = ((1:0.9709036084282487,(8:0.027932027270208215,9:0.027932027270208215):0.9429715811580406):0.029096391571751257,(((2:0.027501276113688248,3:0.027501276113688248):0.37672498232094354,(4:0.24851873157626125,5:0.24851873157626125):0.15570752685837053):0.18528716377119536,(6:0.04157591521208373,7:0.04157591521208373):0.5479375069937434):0.41048657779417286):0.0;
tree STATE_760000 = ((1:0.8869563204643596,(8:0.6119847339913868,9:0.6119847339913868):0.2749715864729728):0.11304367953564043,(((2:0.1436969931878898,3:0.1436969931878898):0.2505635384954607,(4:0.04152304946055113,5:0.04152304946055113):0.3527374822227994):0.4785495394373993,(6:0.05515023524138672,7:0.05515023524138672):0.8176598358793631):0.12718992887925018):0.0;
tree STATE_770000 = ((1:0.9240118815230585,(8:0.0728634173813913,9:0.0728634173813913):0.8511484641416671):0.07598811847694154,(((2:0.39750228325548626,3:0.39750228325548626):0.048819100954935024,(4:0.11357797135591648,5:0.11357797135591648):0.3327434128545048):0.11035902447597906,(6:0.3263872664398021,7:0.3263872664398021):0.23029314224659825):0.44331959131359966):0.0;
tree STATE_780000 = ((1:0.7598191799497915,(8:0.07140290751650208,9:0.07140290751650208):0.6884162724332894):0.24018082005020847,(((2:0.13475559135885282,3:0.13475559135885282):0.15177833298310314,(4:0.07525481547441172,5:0.07525481547441172):0.21127910886754425):0.559335237577943,(6:0.4733545297238758,7:0.4733545297238758):0.3725146321960231):0.15413083808010108):0.0;
tree STATE_790000 = ((1:0.6645854204908167,(8:0.010676946166813995,9:0.010676946166813995):0.6539084743240027):0.3354145795091833,(((2:0.2959452197696672,3:0.2959452197696672):0.0848063807502843,(4:0.09890453374438746,5:0.09890453374438746):0.28184706677556404):0.24296895965573262,(6:0.4044780354221103,7:0.4044780354221103):0.2192425247535738):0.3762794398243159):0.0;
tree STATE_800000 = ((1:0.8108143899988844,(8:0.0022715417569022732,9:0.0022715417569022732):0.8085428482419822):0.18918561000111556,(((2:0.3905687168111473,3:0.3905687168111473):0.006654436971536115,(4:0.015222733810699687,5:0.015222733810699687):0.38200041997198375):0.20531049497668313,(6:0.4506904770808812,7:0.4506904770808812):0.15184317167848532):0.39746635124063345):0.0;
tree STATE_810000 = ((1:0.9442812278766197,(8:0.26345029453725965,9:0.26345029453725965):0.6808309333393601):0.05571877212338028,(((2:0.21865199944934133,3:0.21865199944934133):0.16797841145106363,(4:0.1194078723576986,5:0.1194078723576986):0.26722253854270633):0.19896834685526688,(6:0.4127935621609855,7:0.4127935621609855):0.17280519559468632):0.41440124224432817):0.0;
tree STATE_820000 = ((1:0.8466162176606642,(8:0.1901105608254613,9:0.1901105608254613):0.6565056568352029):0.1533837823393358,(((2:0.2780497498632349,3:0.2780497498632349):0.06431358618877331,(4:0.08233752792951246,5:0.08233752792951246):0.26002580812249576):0.3929728023461836,(6:0.5306406446903572,7:0.5306406446903572):0.20469549370783457):0.2646638616018082):0.0;
tree STATE_830000 = ((1:0.6850239374878695,(8:0.11408894590809057,9:0.11408894590809057):0.5709349915797789):0.3149760625121305,(((2:0.39207365059599175,3:0.39207365059599175):0.06839174261748293,(4:0.17638770276973997,5:0.17638770276973997):0.2840776904437347):0.4018205776495195,(6:0.25428697207381595,7:0.25428697207381595):0.6079989987891783):0.1377140291370058):0.0;
tree STATE_840000 = ((1:0.8628085019753676,(8:0.0328299038122408,9:0.0328299038122408):0.8299785981631268):0.13719149802463237,(((2:0.4316486627875384,3:0.4316486627875384):0.23355009434753066,(4:0.18079749355502814,5:0.18079749355502814):0.4844012635800409):0.1334352700915309,(6:0.08601221865023752,7:0.08601221865023752):0.7126218085763624):0.20136597277340007):0.0;
tree STATE_850000 = ((1:0.8443102326241894,(8:0.6437213162811144,9:0.6437213162811144):0.20058891634307496):0.15568976737581064,(((2:0.09410818970901806,3:0.09410818970901806):0.2937897818909087,(4:0.059963641788171276,5:0.059963641788171276):0.3279343298117555):0.3814581687154942,(6:0.7656289887317334,7:0.7656289887317334):0.003727151583687638):0.230643859684579):0.0;
tree STATE_860000 = ((1:0.5438378622755424,(8:0.13136811159516076,9:0.13136811159516076):0.41246975068038166):0.4561621377244576,(((2:0.3001311976688347,3:0.3001311976688347):0.03701225818690895,(4:0.12609620333083071,5:0.12609620333083071):0.21104725252491296):0.34096114887565165,(6:0.18950591009809417,7:0.18950591009809417):0.4885986946333012):0.32189539526860467):0.0;
tree STATE_870000 = ((1:0.9111863354540434,(8:0.09052303180334999,9:0.09052303180334999):0.8206633036506934):0.08881366454595663,(((2:0.21811541239795312,3:0.21811541239795312):0.16344325937858595,(4:0.3267535944430903,5:0.3267535944430903):0.05480507733344875):0.2973962700056754,(6:0.16809690268711125,7:0.16809690268711125):0.5108580390951032):0.3210450582177855):0.0;
tree STATE_880000 = ((1:0.6703038667639316,(8:0.0012659881514864096,9:0.0012659881514864096):0.6690378786124452):0.3296961332360684,(((2:0.16928692730153067,3:0.16928692730153067):0.1464221254987857,(4:0.02479561858936376,5:0.02479561858936376):0.2909134342109526):0.24763452439975353,(6:0.4049076003769905,7:0.4049076003769905):0.15843597682307942):0.4366564227999301):0.0;
tree STATE_890000 = ((1:0.6318559418455559,(8:0.3104064193898443,9:0.3104064193898443):0.3214495224557116):0.36814405815444406,(((2:0.3939202732723856,3:0.3939202732723856):0.18371508336784032,(4:0.12442402922199076,5:0.12442402922199076):0.45321132741823517):0.13825729973219303,(6:0.5560081077508102,7:0.5560081077508102):0.15988454862160872):0.28410734362758105):0.0;
tree STATE_900000 = ((1:0.8020102847973204,(8:0.22561214652914205,9:0.22561214652914205):0.5763981382681783):0.1979897152026796,(((2:0.34180087367894757,3:0.34180087367894757):0.05407117824694313,(4:0.01578475651648112,5:0.01578475651648112):0.38008729540940955):0.3260503446258204,(6:0.3755164757784437,7:0.3755164757784437):0.3464059207732674):0.2780776034482889):0.0;
tree STATE_910000 = ((1:0.8089459597926841,(8:0.1437452844038653,9:0.1437452844038653):0.6652006753888189):0.19105404020731587,(((2:0.24337250096520102,3:0.24337250096520102):0.057929331632286085,(4:0.10376588597055299,5:0.10376588597055299):0.19753594662693413):0.48321491212651446,(6:0.16152267141618026,7:0.16152267141618026):0.6229940733078213):0.21548325527599843):0.0;
tree STATE_920000 = ((1:0.840910338605379,(8:0.16452829963061055,9:0.16452829963061055):0.6763820389747685):0.159089661394621,(((2:0.2567244048123589,3:0.2567244048123589):0.2534824435730636,(4:0.024826318740216458,5:0.024826318740216458):0.48538052964520606):0.1322181016280869,(6:0.05304609489430879,7:0.05304609489430879):0.5893788551192006):0.3575750499864906):0.0;
tree STATE_930000 = ((1:0.6310842451626817,(8:0.0954474718110267,9:0.0954474718110267):0.5356367733516549):0.3689157548373183,(((2:0.21911571951967393,3:0.21911571951967393):0.02614139617960895,(4:0.010450341857071457,5:0.010450341857071457):0.23480677384221144):0.29987063439016803,(6:0.20846323405652054,7:0.20846323405652054):0.33666451603293035):0.4548722499105491):0.0;
tree STATE_940000 = ((1:0.6282734353434022,(8:0.033403151083458565,9:0.033403151083458565):0.5948702842599436):0.3717265646565978,(((2:0.24004245458142656,3:0.24004245458142656):0.1984150104510465,(4:0.07393164726855608,5:0.07393164726855608):0.364525817763917):0.359676895492278,(6:0.26577083573637295,7:0.26577083573637295):0.5323635247883781):0.20186563947524894):0.0;
tree STATE_950000 = ((1:0.767276926169993,(8:0.17879323878329983,9:0.17879323878329983):0.5884836873866932):0.232723073830007,(((2:0.10543259256812383,3:0.10543259256812383):0.20795659259730842,(4:0.07103198529208657,5:0.07103198529208657):0.24235719987334567):0.23890274647766852,(6:0.19897283382738185,7:0.19897283382738185):0.3533190978157189):0.44770806835689925):0.0;
tree STATE_960000 = ((1:0.7295856072850858,(8:0.26302298661508905,9:0.26302298661508905):0.46656262066999676):0.2704143927149142,(((2:0.052723933557286384,3:0.052723933557286384):0.19049927165972155,(4:0.1466852632464174,5:0.1466852632464174):0.09653794197059054):0.34494939439117955,(6:0.15771495805172286,7:0.15771495805172286):0.4304576415564646):0.41182740039181254):0.0;
tree STATE_970000 = ((1:0.9029174607734979,(8:0.002587100186902833,9:0.002587100186902833):0.9003303605865951):0.09708253922650212,(((2:0.3370578549720971,3:0.3370578549720971):0.010413765999334246,(4:0.06676384176863129,5:0.06676384176863129):0.2807077792028001):0.3606021946808845,(6:0.13782506948222034,7:0.13782506948222034):0.5702487461700956):0.29192618434768414):0.0;
tree STATE_980000 = ((1:0.8492878699211391,(8:0.1536927533830289,9:0.1536927533830289):0.6955951165381102):0.15071213007886086,(((2:0.13787577280500343,3:0.13787577280500343):0.4948267711950728,(4:0.09481250912855212,5:0.09481250912855212):0.5378900348715241):0.13562211346796715,(6:0.5802584914158919,7:0.5802584914158919):0.18806616605215143):0.23167534253195665):0.0;
tree STATE_990000 = ((1:0.9182525051876228,(8:0.30607259332850045,9:0.30607259332850045):0.6121799118591223):0.0817474948123772,(((2:0.1590902189714553,3:0.1590902189714553):0.3532326835641502,(4:0.08162478551802135,5:0.08162478551802135):0.4306981170175842):0.1991240047919125,(6:0.09348035008365312,7:0.09348035008365312):0.617966557243865):0.28855309267248197):0.0;
tree STATE_1000000 = ((1:0.8628830182385026,(8:0.22395538306908727,9:0.22395538306908727):0.6389276351694153):0.13711698176149745,(((2:0.13896255223842285,3:0.13896255223842285):0.07368924432164714,(4:0.049581040219699474,5:0.049581040219699474):0.16307075634037052):0.25236202579590405,(6:0.18152026225563933,7:0.18152026225563933):0.2834935601003347):0.5349861776440259):0.0;
End;
"""
    )
    assert(res == "((10_0112212221:0.8135771558123348,(8_2120010021:0.15141414803270203,9_2120010021:0.15141414803270203):0.6621630077796328):0.18642284418766525,(((1_2012212021:0.2189736340872589,2_2112212021:0.2189736340872589):0.1492781153547722,(3_2112212021:0.09923796583327271,4_2112212021:0.09923796583327271):0.2690137836087584):0.29969688182777016,(5_0012212221:0.2864666681876864,6_0012012221:0.2864666681876864):0.38148196308211485):0.33205136873019875);")


def _get_sorted_observed_mutations(cm: pd.DataFrame) -> List[int]:
    cm_np = cm.to_numpy()
    sorted_observed_mutations = sorted(
        list(
            set(
                [
                    cm_np[i, j]
                    for i in range(cm_np.shape[0])
                    for j in range(cm_np.shape[1])
                    if (cm_np[i, j] >= 1)
                ]
            )
        )
    )
    return sorted_observed_mutations


def test__get_sorted_observed_mutations():
    cm_dict_leaves = {
        "3": [10, 2, 0, 0, 0, 0, 0, 0, 0, -1],
        "4": [10, 0, 3, 0, 0, 0, 0, 0, 0, -1],
        "5": [0, 0, 0, 0, 5, 6, 7, 0, 0, -1],
        "6": [0, 0, 0, 4, 0, 6, 0, 8, 9, -1],
    }
    cm = pd.DataFrame(cm_dict_leaves).T
    assert(
        _get_sorted_observed_mutations(cm) == [
            2, 3, 4, 5, 6, 7, 8, 9, 10
        ]
    )


def _relabel_states(cm: pd.DataFrame) -> pd.DataFrame:
    """
    Relabel states so that they go from 1 to number_of_observed_mutations.
    Missing state will further map to number_of_observed_mutations + 1 (since this is how TiDeTree encodes missing data).
    See test__relabel_states for an example.
    """
    sorted_observed_mutations = _get_sorted_observed_mutations(cm)
    new_character_matrix = cm
    for new_idx, observed_mutation in enumerate(sorted_observed_mutations):
        # Note that because we are going in increasing order it is not possible to mess up
        # the relabeling.
        new_character_matrix[new_character_matrix == observed_mutation] = new_idx + 1
    new_character_matrix[new_character_matrix < 0] = len(sorted_observed_mutations) + 1
    return new_character_matrix


def test__relabel_states():
    cm_dict_leaves = {
        "3": [10, 2, 0, 0, 0, 0, 0, 0, 0, -1],
        "4": [10, 0, 3, 0, 0, 0, 0, 0, 0, -1],
        "5": [0, 0, 0, 0, 5, 6, 7, 0, 0, -1],
        "6": [0, 0, 0, 4, 0, 6, 0, 8, 9, -1],
    }
    cm = pd.DataFrame(cm_dict_leaves).T
    pd.testing.assert_frame_equal(
        _relabel_states(cm),
        pd.DataFrame(
            {
                "3": [9, 1, 0, 0, 0, 0, 0, 0, 0, 10],
                "4": [9, 0, 2, 0, 0, 0, 0, 0, 0, 10],
                "5": [0, 0, 0, 0, 4, 5, 6, 0, 0, 10],
                "6": [0, 0, 0, 3, 0, 5, 0, 7, 8, 10],
            }
        ).T
    )


class TiDeTree(BranchLengthEstimator):
    def __init__(
        self,
        priors: Optional[Dict[int, float]],  # Probability of each state ("q" distribution in Cassiopeia/LAML)
        experiment_duration: float = 54.0,
        edit_duration: float = 54.0,
        chain_length: int = 1000000,
        random_seed: int = 42,
        subsampling_probability: float = 1.0,  # Extant sampling probability (rho in TiDeTree)
        silencing_rate: float = 0.0,
    ):
        self.experiment_duration = experiment_duration
        self.edit_duration = edit_duration
        self.priors = priors
        self.chain_length = chain_length
        self.random_seed = random_seed
        self.subsampling_probability = subsampling_probability
        self.silencing_rate = silencing_rate

        # Download TiDeTree if not present
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.dir_path = dir_path
        tidetree_bin_path = os.path.join(dir_path, "tidetree.jar")
        self.tidetree_bin_path = tidetree_bin_path
        if not os.path.exists(tidetree_bin_path):
            print(f"Going to get tidetree.jar...")
            wget.download(
                "https://github.com/seidels/tidetree/raw/main/bin/tidetree.jar",
                f"{dir_path}/"
            )
            os.chmod(tidetree_bin_path, 0o444)
            print(f"Done!")

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        tree_original = tree  # Because we are going to remap states in the cm, etc.
        tree = deepcopy(tree)
        experiment_duration = self.experiment_duration
        edit_duration = self.edit_duration
        dir_path = self.dir_path
        priors = self.priors
        chain_length = self.chain_length
        random_seed = self.random_seed
        subsampling_probability = self.subsampling_probability
        silencing_rate = self.silencing_rate

        # If priors is None, then we set the prior to be uniform over all observed states.
        # We further map all states to the range [1, number_of_observed_mutations] because
        # TiDeTree assumes states are numbered that way.
        if priors is None:
            tree.character_matrix = _relabel_states(tree.character_matrix)
            number_of_observed_mutations = len(_get_sorted_observed_mutations(tree.character_matrix))
            priors = {
                i: 1.0 / number_of_observed_mutations
                for i in range(1, number_of_observed_mutations + 1)
            }


        # Need to initialize branch lengths.
        # I will use the number of mutations.
        tree = maximum_parsimony(tree)
        initialization_ble = NumberOfMutationsBLE(length_of_mutationless_edges=0.5, make_ultrametric=True)
        initialization_ble.estimate_branch_lengths(tree)
        # I now need to scale the tree. Since no edits happen past the edit_duration,
        # I will scale to edit_duration, then extend the leaves to the experiment_duration
        tree.scale_to_unit_length(depth=edit_duration)
        times_original = tree.get_times()
        times_new = times_original.copy()
        for leaf_name in tree.leaves:
            times_new[leaf_name] = experiment_duration
        tree.set_times(times_new)

        tidetree_bin_path = self.tidetree_bin_path
        original_nodes = sorted(list(tree.nodes))

        xml_str = _get_xml_str(
            tree=tree,
            experiment_duration=experiment_duration,
            edit_duration=edit_duration,
            priors=priors,
            chain_length=chain_length,
            subsampling_probability=subsampling_probability,
            silencing_rate=silencing_rate,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            # tmp_dir = os.path.join(dir_path, "tmp_dir")
            # Write xml
            with open(f"{tmp_dir}/example.xml", "w") as f:
                f.write(xml_str)
            # Run optimization
            command = f"{JAVA} -jar {tidetree_bin_path} -seed {random_seed} -overwrite {tmp_dir}/example.xml > {tmp_dir}/tidetree.out"

            with pushd(tmp_dir):
                # I pushd because TiDeTree writes the output trees to the current working directory
                print(f"Going to run TiDeTree command: {command}")
                subprocess.run(command, shell=True)
                # Read trees and create posterior mean.
                try:
                    with open(os.path.join(tmp_dir, f"example.tree.{random_seed}.trees"), "r") as tree_file:
                        nwk = get_mean_newick_str(tree_file.read())
                except FileNotFoundError:
                    raise TiDeTreeError(f"TiDeTree failed. Command: {command}")

        # Now post-process.
        tree_with_bls_but_wrong_internal_node_names = deepcopy(tree)
        tree_with_bls_but_wrong_internal_node_names.populate_tree(tree=nwk)

        # Now we need to create the mapping between correct and wrong node names
        # The leaf names will agree
        correct_to_wrong_node_name = {leaf: leaf for leaf in tree.leaves}
        for node in tree.depth_first_traverse_nodes(postorder=True):
            if node != tree.root:
                # Set the parent
                correct_to_wrong_node_name[tree.parent(node)] = tree_with_bls_but_wrong_internal_node_names.parent(correct_to_wrong_node_name[node])

        # Now we set the times.
        times = tree_with_bls_but_wrong_internal_node_names.get_times()
        times = {
            node: times[correct_to_wrong_node_name[node]]
            for node in tree.nodes
        }
        tree_original.set_times(times)
        assert(
            sorted(list(tree.nodes)) == original_nodes
        )


def test_TiDeTree_example():
    tree = nx.DiGraph()
    tree = CassiopeiaTree(tree="((((1_2012212021:51,2_2112212021:51):1,(3_2112212021:51,4_2112212021:51):1):1,(5_0012212221:52,6_0012012221:52):1):1,((8_2120010021:52,9_2120010021:52):1,10_0112212221:53):1);")
    cm_dict = {
        "1_2012212021": [2,0,1,2,2,1,2,0,2,1],
        "2_2112212021": [2,1,1,2,2,1,2,0,2,1],
        "3_2112212021": [2,1,1,2,2,1,2,0,2,1],
        "4_2112212021": [2,1,1,2,2,1,2,0,2,1],
        "5_0012212221": [0,0,1,2,2,1,2,2,2,1],
        "6_0012012221": [0,0,1,2,0,1,2,2,2,1],
        "8_2120010021": [2,1,2,0,0,1,0,0,2,1],
        "9_2120010021": [2,1,2,0,0,1,0,0,2,1],
        "10_0112212221": [0,1,1,2,2,1,2,2,2,1],
    }
    tree.set_character_states_at_leaves(
        cm_dict
    )
    tree.character_matrix = pd.DataFrame(cm_dict).T
    model = TiDeTree(
        priors={1: 0.9, 2: 0.1},
        experiment_duration=54.0,
        edit_duration=36.0,
    )
    model.estimate_branch_lengths(tree)
    np.testing.assert_almost_equal(
        tree.get_branch_length(tree.parent("1_2012212021"), "1_2012212021"), 22.948821331557898, decimal=0
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length(tree.parent("2_2112212021"), "2_2112212021"), 22.948821331557898, decimal=0
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length(tree.parent("3_2112212021"), "3_2112212021"), 7.202817446581342, decimal=0
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length(tree.parent("3_2112212021"), "3_2112212021"), 7.202817446581342, decimal=0
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length(tree.parent("9_2120010021"), "9_2120010021"), 6.615981859357156, decimal=0
    )


def test_TiDeTree():
    tree = nx.DiGraph()
    tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"]),
    tree.add_edges_from(
        [
            ("0", "1"),
            ("0", "2"),
            ("1", "3"),
            ("1", "4"),
            ("2", "5"),
            ("2", "6"),
        ]
    )
    tree = CassiopeiaTree(tree=tree)
    cm_dict = {
        "0": [0, 0, 0, 0, 0, 0, 0, 0, 0],#, 0],
        "1": [1, 0, 0, 0, 0, 0, 0, 0, 0],#, -1],
        "2": [0, 0, 0, 0, 0, 6, 0, 0, 0],#, -1],
        "3": [1, 2, 0, 0, 0, 0, 0, 0, 0],#, -1],
        "4": [1, 0, 3, 0, 0, 0, 0, 0, 0],#, -1],
        "5": [0, 0, 0, 0, 5, 6, 7, 0, 0],#, -1],
        "6": [0, 0, 0, 4, 0, 6, 0, 8, 9],#, -1],
    }
    cm_dict_leaves = {
        "3": [1, 2, 0, 0, 0, 0, 0, 0, 0],#, -1],
        "4": [1, 0, 3, 0, 0, 0, 0, 0, 0],#, -1],
        "5": [0, 0, 0, 0, 5, 6, 7, 0, 0],#, -1],
        "6": [0, 0, 0, 4, 0, 6, 0, 8, 9],#, -1],
    }
    tree.set_all_character_states(
        cm_dict
    )
    tree.character_matrix = pd.DataFrame(cm_dict_leaves).T
    model = TiDeTree(
        # priors={i: 1.0/9.0 for i in range(1, 10)}  # Passing in None has the same effect so we try that instead.
        priors=None,
    )
    model.estimate_branch_lengths(tree)
    tree.scale_to_unit_length()
    np.testing.assert_almost_equal(
        tree.get_branch_length("0", "1"), 0.6757221911820785, decimal=1
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length("0", "2"), 0.37090443169941606, decimal=1
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length("1", "3"), 0.32427780881792145, decimal=1
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length("1", "4"), 0.32427780881792145, decimal=1
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length("2", "5"), 0.6290955683005839, decimal=1
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length("2", "6"), 0.6290955683005839, decimal=1
    )

    # Just gonna chech ConvexML here.
    model = IIDExponentialMLE(
        minimum_branch_length=0.01,
        pseudo_mutations_per_edge=0.1,
        pseudo_non_mutations_per_edge=0.1,
    )
    model.estimate_branch_lengths(tree)
    tree.scale_to_unit_length()
    np.testing.assert_almost_equal(
        tree.get_branch_length("0", "1"), 0.5315571567058911, decimal=3
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length("0", "2"), 0.23064542572443283, decimal=3
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length("1", "3"), 0.46844284329402397, decimal=3
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length("1", "4"), 0.46844284329402397, decimal=3
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length("2", "5"), 0.7693545742755599, decimal=3
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length("2", "6"), 0.7693545742755599, decimal=3
    )


def test_TiDeTree_missing_data():
    tree = nx.DiGraph()
    tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"]),
    tree.add_edges_from(
        [
            ("0", "1"),
            ("0", "2"),
            ("1", "3"),
            ("1", "4"),
            ("2", "5"),
            ("2", "6"),
        ]
    )
    tree = CassiopeiaTree(tree=tree)
    cm_dict = {
        "0": [0, 0, 0, 0, 0, 0, 0, 0, 0],#, 0],
        "1": [-1, 0, 0, 0, 0, 0, 0, 0, 0],#, -1],
        "2": [0, 0, 0, 0, 0, 6, 0, 0, 0],#, -1],
        "3": [-1, 2, 0, 0, 0, 0, 0, 0, 0],#, -1],
        "4": [-1, 0, 3, 0, 0, 0, 0, 0, 0],#, -1],
        "5": [0, 0, 0, 0, 5, 6, 7, 0, 0],#, -1],
        "6": [0, 0, 0, 4, 0, 6, 0, 8, 9],#, -1],
    }
    cm_dict_leaves = {
        "3": [-1, 2, 0, 0, 0, 0, 0, 0, 0],#, -1],
        "4": [-1, 0, 3, 0, 0, 0, 0, 0, 0],#, -1],
        "5": [0, 0, 0, 0, 5, 6, 7, 0, 0],#, -1],
        "6": [0, 0, 0, 4, 0, 6, 0, 8, 9],#, -1],
    }
    tree.set_all_character_states(
        cm_dict
    )
    tree.character_matrix = pd.DataFrame(cm_dict_leaves).T
    model = TiDeTree(
        priors=None,
    )
    model.estimate_branch_lengths(tree)
    tree.scale_to_unit_length()
    np.testing.assert_almost_equal(
        tree.get_branch_length("0", "1"), 0.6757221911820785, decimal=1
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length("0", "2"), 0.3684018781505094, decimal=1
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length("1", "3"), 0.32427780881792145, decimal=1
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length("1", "4"), 0.32427780881792145, decimal=1
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length("2", "5"), 0.6315981218494906, decimal=1
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length("2", "6"), 0.6315981218494906, decimal=1
    )

    # Just gonna chech ConvexML here.
    model = IIDExponentialMLE(
        minimum_branch_length=0.01,
        pseudo_mutations_per_edge=0.1,
        pseudo_non_mutations_per_edge=0.1,
    )
    model.estimate_branch_lengths(tree)
    tree.scale_to_unit_length()
    np.testing.assert_almost_equal(
        tree.get_branch_length("0", "1"), 0.21198768482339514, decimal=3
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length("0", "2"), 0.21939500829399317, decimal=3
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length("1", "3"), 0.7880123151766049, decimal=3
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length("1", "4"), 0.7880123151766049, decimal=3
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length("2", "5"), 0.7806049916933218, decimal=3
    )
    np.testing.assert_almost_equal(
        tree.get_branch_length("2", "6"), 0.7806049916933218, decimal=3
    )


def test_TiDeTree_missing_data_minimal():
    tree = nx.DiGraph()
    tree.add_nodes_from(["0", "1", "2"]),
    tree.add_edges_from(
        [
            ("0", "1"),
            ("0", "2"),
        ]
    )
    tree = CassiopeiaTree(tree=tree)
    cm_dict = {
        "0": [0],
        "1": [0],
        "2": [-1],
    }
    cm_dict_leaves = {
        "1": [0],
        "2": [-1],
    }
    tree.set_all_character_states(
        cm_dict
    )
    tree.character_matrix = pd.DataFrame(cm_dict_leaves).T
    model = TiDeTree(
        priors=None,
    )
    model.estimate_branch_lengths(tree)
    tree.scale_to_unit_length()
    np.testing.assert_almost_equal(
        tree.get_branch_length("0", "1"), 1.0, decimal=3
    )

    # Just gonna chech ConvexML here.
    model = IIDExponentialMLE(
        minimum_branch_length=0.01,
        pseudo_mutations_per_edge=0.1,
        pseudo_non_mutations_per_edge=0.1,
    )
    model.estimate_branch_lengths(tree)
    tree.scale_to_unit_length()
    np.testing.assert_almost_equal(
        tree.get_branch_length("0", "1"), 1.0, decimal=3
    )
