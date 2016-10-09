# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 01:25:05 2016

@author: julmaud
"""

import case_rec

#cases_as_docs, docs_per_case = case_rec.format_data_ads()
#cases_as_docs, docs_per_case = case_rec.format_data_bible()


es = case_rec.setup_es()

#case_rec.build_elasticsearch_database(es, cases_as_docs, docs_per_case)

"""
for case_id in cases_as_docs.keys():
    print('#######')
    print('case_id: {}'.format(case_id))
    ranked_patterns = case_rec.get_ranked_patterns(es, case_id, 5)
    print('patterns :{}'.format(ranked_patterns))
    rk = case_rec.get_ranked_cases(es, ranked_patterns)
    print('ranked cases: {}'.format(rk))
print('#######')
"""

def demo_case_rank(es):
    print('Connecting to the database')
    es.indices.delete(index = 'documents')
    es.indices.delete(index = 'cases')
    cases_as_docs, docs_per_case = case_rec.format_data()
    case_rec.build_elasticsearch_database(es, cases_as_docs, docs_per_case)

    pattern = input('Enter a pattern using \' \': ')
    print('')
    print('Currently ranking cases...')
    print('')
    rk = case_rec.get_ranked_cases(es, pattern, 5)
    print('Cases are ranked based on the pattern: {}'.format(pattern))
    for (case, score) in rk:
        print('{} : {}'.format(case,score))
    print('')
    case_id = input('Enter a case_id using \' \': ')
    rk_file = case_rec.get_ranked_files(es, pattern, case_id, 10)
    print('')
    print('ranked files of case {} based on pattern: {}'.format(case_id, pattern))
    for fil in rk_file:
        print(fil)

def demo_pattern_finding(es):
    es.indices.delete(index = 'documents')
    es.indices.delete(index = 'cases')
    cases_as_docs, docs_per_case = case_rec.format_data()
    case_rec.build_elasticsearch_database(es, cases_as_docs, docs_per_case)
    
    case_id = input('Enter a case_id using \' \': ')
    print('')
    rk = case_rec.get_ranked_patterns(es, case_id, 10)
    for patt,score in rk:
        print('{} : {}'.format(patt, score))

#demo_case_rank(es)