"""Structural checks of committed support-aware history against declared exposure."""
import math


def validate_supported_history(history, *, epochs, records, sample_ids, batch_sizes,
                               validation_epochs, validation_population, weights,
                               expected_lrs):
    """Check internal consistency; metric values and declared support are not remeasured."""
    def require(condition, detail):
        if not condition:
            raise ValueError(f'support-aware history mismatch: {detail}')

    require(len(records) == epochs * len(batch_sizes), 'optimizer steps')
    fields = {'lr'} if epochs else set()
    require(history.get('lr', []) == expected_lrs, 'learning-rate rows')
    train_support = {name: [] for name in (*weights, 'total')}
    admitted = set(sample_ids)
    for epoch in range(epochs):
        rows = records[epoch * len(batch_sizes):(epoch + 1) * len(batch_sizes)]
        seen = []
        for row, size in zip(rows, batch_sizes, strict=True):
            require(len(row['sample_ids']) == size, 'batch population')
            seen.extend(row['sample_ids'])
        require(len(seen) == len(set(seen)) and set(seen) <= admitted, 'epoch sample membership')
        for name in weights:
            train_support[name].append(sum(row['support'][name] for row in rows))
        train_support['total'].append(len(seen))

    for partition, schedule in (('train', list(range(1, epochs + 1))), ('val', validation_epochs)):
        if not schedule:
            continue
        population = train_support['total'] if partition == 'train' else [validation_population] * len(schedule)
        for name in (*weights, 'total'):
            support_key = f'{partition}_support_{name}'
            fields.add(support_key)
            counts = history.get(support_key)
            require(isinstance(counts, list) and len(counts) == len(schedule), support_key)
            require(all(type(count) is int and 0 <= count <= size
                        for count, size in zip(counts, population, strict=True)), support_key)
            if partition == 'train':
                require(counts == train_support[name], f'{support_key} versus exposure')
            elif name == 'total':
                require(counts == population, support_key)
            available = [epoch for epoch, count in zip(schedule, counts, strict=True) if count > 0]
            if available:
                metric, indices = f'{partition}_{name}', f'{partition}_{name}_epoch'
                fields.update((metric, indices))
                require(history.get(indices) == available
                        and all(type(epoch) is int for epoch in history[indices]), indices)
                values = history.get(metric)
                require(isinstance(values, list) and len(values) == len(available), metric)
                require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                            for value in values), metric)
    require(set(history) == fields, 'missing or unexpected fields')
