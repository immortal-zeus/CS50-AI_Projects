import os
import random
import re
import sys
from copy import deepcopy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob_dis = dict()
    for_all_pages = (1 - damping_factor) /len(corpus)
    if len(corpus[page]) != 0:
        for_linked_pages = (damping_factor / len(corpus[page])) + for_all_pages
        for all_pages in corpus:
            if all_pages not in corpus[page]:
                prob_dis[all_pages]= for_all_pages
            else:
                prob_dis[all_pages] = for_linked_pages
    else:
        for all_pages in corpus:
            prob_dis[all_pages]= for_all_pages
    return prob_dis

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    final_dict={k:0 for k, i in corpus.items()}
    ran_page = random.choice(list(corpus.keys()))
    sample_dict = transition_model(corpus,ran_page,damping_factor)
    final_dict[ran_page]+=1
    for o in range(n):
        ran_page = random.choices(list(sample_dict.keys()),weights=list(sample_dict.values()))[0]
        final_dict[ran_page]+=1
        sample_dict = transition_model(corpus,ran_page,damping_factor)
    final_dict = {k: i/n for k, i in final_dict.items()}
    return final_dict

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    final_dict={k : 1/len(corpus.keys()) for k in list(corpus.keys())}
    while True:
        instance_dict = deepcopy(final_dict)
        for key in list(corpus.keys()):
            final_dict[key]= (1-damping_factor)/len(corpus.keys())
            all_links = [li for li in list(corpus.keys()) if key in corpus[li]]
            sum_all_links=0
            if all_links:
                for i in all_links:
                    sum_all_links+=instance_dict[i] / len(corpus[i])
            final_dict[key]+=damping_factor*sum_all_links
        if abs(instance_dict[key] - final_dict[key]) <=0.001:
            final_dict = {eys : ules/sum(final_dict.values()) for eys, ules in final_dict.items()}
            return final_dict

if __name__ == "__main__":
    main()
