#python new_snv_script.py -i Test_data_local/cae_pe_test/group_pe_test_candidate_mutation_table.npz -c  Test_data_local/cae_pe_test/group_pe_test_coverage_matrix_raw.npz -r reference_genomes/Cae_ref -o cae_accusnv_pe_local

python accusnv_downstream.py -i  Test_data_local/candidate_mutation_table_final.npz -r reference_genomes/Cae_ref -o cae_accusnv_ds_pe_downstream

#python new_snv_script.py -i Test_data_local/cae_se_test/group_se_test_candidate_mutation_table.npz -c  Test_data_local/cae_se_test/group_se_test_coverage_matrix_raw.npz -r reference_genomes/Cae_ref -o cae_accusnv_se_local

