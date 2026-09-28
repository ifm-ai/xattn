"""Validate complete pytest shard coverage or self-benchmark result manifests."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def correctness(reports):
    first = reports[0]
    canonical = first['canonical_nodeids']
    if canonical != sorted(set(canonical)):
        raise ValueError('canonical node IDs must be sorted and unique')
    count = first['shard_count']
    if sorted(r['shard_index'] for r in reports) != list(range(count)):
        raise ValueError('missing or duplicate shards')
    observed = []
    totals = Counter()
    for report in reports:
        if (report['shard_count'] != count or report['canonical_nodeids'] != canonical
                or report['coverage_sha256'] != digest(canonical)):
            raise ValueError('inconsistent canonical coverage')
        expected = [node for node in canonical
                    if int.from_bytes(hashlib.sha256(node.encode()).digest()[:8], 'big') % count == report['shard_index']]
        if sorted(report['selected_nodeids']) != expected:
            raise ValueError('incorrect shard partition')
        observed += report['selected_nodeids']
        xml = ET.parse(report['junit']).getroot()
        cases = xml.findall('.//testcase')
        if len(cases) != len(expected):
            raise ValueError('pytest results do not cover every selected node')
        totals['tests'] += len(cases)
        totals['failures'] += sum(case.find('failure') is not None for case in cases)
        totals['errors'] += sum(case.find('error') is not None for case in cases)
        totals['skipped'] += sum(case.find('skipped') is not None for case in cases)
        totals['failed_processes'] += int(report['exit_code'] != 0)
    if Counter(observed) != Counter(canonical):
        raise ValueError('shard union differs from canonical coverage')
    return dict(totals, passed=not any(totals[k] for k in ('failures','errors','failed_processes')))


def speed(reports):
    count = 0
    for report in reports:
        manifest = report['manifest']
        encoded = json.dumps(manifest, sort_keys=True, separators=(',', ':'))
        if hashlib.sha256(encoded.encode()).hexdigest() != report['coverage_sha256']:
            raise ValueError('benchmark manifest hash mismatch')
        records = report['records']
        keys = tuple(manifest[0])
        def identity(record):
            return json.dumps({key:record[key] for key in keys}, sort_keys=True)
        if len(set(map(identity,manifest)))!=len(manifest) or Counter(map(identity,records))!=Counter(map(identity,manifest)):
            raise ValueError('benchmark records incomplete or duplicated')
        for record in records:
            if record.get('correctness') != 'passed_sampled_fp64' or 'error' in record:
                raise ValueError('benchmark correctness did not pass')
            for direction in ('fwd','bwd'):
                if record[direction]['latency_us'] <= 0 or not record[direction]['samples_us']:
                    raise ValueError('missing valid timing')
        count += len(records)
    return {'passed':True,'records':count}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('correctness','speed'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('reports',nargs='+',type=Path)
    args=parser.parse_args()
    try:
        reports=[json.loads(p.read_text()) for p in args.reports]
        result=(correctness if args.kind=='correctness' else speed)(reports)
    except Exception as exc:
        result={'passed':False,'error':repr(exc)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':
    main()
