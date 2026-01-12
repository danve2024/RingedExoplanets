from nested_sampling import ns

chunks = 14
splits = 5
offset = 4

for i in range(4, splits):
    ns.kfold_split(chunks, [i + offset], f'kfolds/{i}/kfold_observations.jpg')
    ns.update(f'loglikes/kfold{i}.json')
    results = ns.run()
    ns.save(f'kfolds/{i}/nested_sampling_result.npz')
    ns.analyze(f'kfolds/{i}/')
