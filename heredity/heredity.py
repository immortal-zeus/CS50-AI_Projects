import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    final_per=0
    joint_prob_all=1
    trait=None
    for per in list(people.keys()):
        mother = people[per]['mother']
        father= people[per]['father']
        prob_parent = {mother:0, father:0}
        if mother == None and father == None:
            if per in one_gene:
                trait=1
                final_per=PROBS["gene"][1]
            elif per in two_genes:
                trait=2
                final_per=PROBS["gene"][2]
            else:
                trait=0
                final_per=PROBS["gene"][0]
        else:
            for parent in list(prob_parent.keys()):
                if parent in one_gene:
                    prob_parent[parent]=(1-PROBS["mutation"])*0.5
                elif parent in two_genes:
                    prob_parent[parent]=1-PROBS["mutation"]
                else:
                    prob_parent[parent]= PROBS["mutation"]
            if per in one_gene:
                trait=1
                final_per=(prob_parent[mother]*(1-prob_parent[father]))+(prob_parent[father]*(1-prob_parent[mother]))
            elif per in two_genes:
                trait=2
                final_per= prob_parent[mother]*prob_parent[father]
            else:
                trait=0
                final_per= (1-prob_parent[mother])*(1-prob_parent[father])
        if per in have_trait:
            final_per*=PROBS["trait"][trait][True]
        else:
            final_per *= PROBS["trait"][trait][False]
        joint_prob_all*=final_per
    return joint_prob_all

def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for people in list(probabilities.keys()):
        if people in one_gene:
            probabilities[people]["gene"][1]+=p
            if people in have_trait:
                probabilities[people]["trait"][True]+=p
            else:
                probabilities[people]["trait"][False]+=p
        elif people in two_genes:
            probabilities[people]["gene"][2]+=p
            if people in have_trait:
                probabilities[people]["trait"][True]+=p
            else:
                probabilities[people]["trait"][False]+=p
        else:
            probabilities[people]["gene"][0]+=p
            if people in have_trait:
                probabilities[people]["trait"][True]+=p
            else:
                probabilities[people]["trait"][False]+=p

def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for people in list(probabilities.keys()):
        sum = 0
        for val in probabilities[people]["gene"]:
            sum +=probabilities[people]["gene"][val]
        for val in probabilities[people]["gene"]:
            probabilities[people]["gene"][val]/=sum
        sum = 0
        for val in probabilities[people]["trait"]:
            sum += probabilities[people]["trait"][val]
        for val in probabilities[people]["trait"]:
            probabilities[people]["trait"][val]/=sum

if __name__ == "__main__":
    main()
