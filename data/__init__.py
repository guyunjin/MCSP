
benchmarks = {
     'IEMOCAP-DA':{
        'labels':['oth', 'ap', 'o', 'g', 's', 'a', 'b', 'c', 'ans', 'q', 'ag', 'dag'],
        'max_seq_lengths': {
                'text': 44,
                'video': 230,
                'audio': 380
            },
            'feat_dims': {
                'text': 768,
                'video': 1024,
                'audio': 768
            },
    },
    'MIntRec':{
        'labels': [
                    'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize',
                    'Agree', 'Taunt', 'Flaunt',
                    'Joke', 'Oppose',
                    'Comfort', 'Care', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave',
                    'Prevent', 'Greet', 'Ask for help'
        ],
        'max_seq_lengths': {
            'text': 30,
            'video': 230,
            'audio': 480
        },
        'feat_dims': {
            'text': 768,
            'video': 1024,
            'audio': 768
        },

    },
    'MIntRec2.0': {
        'labels': [
            'Acknowledge', 'Advise', 'Agree', 'Apologise', 'Arrange',
            'Ask for help', 'Asking for opinions', 'Care', 'Comfort', 'Complain',
            'Confirm', 'Criticize', 'Doubt', 'Emphasize', 'Explain',
            'Flaunt', 'Greet', 'Inform', 'Introduce', 'Invite',
            'Joke', 'Leave', 'Oppose', 'Plan', 'Praise',
            'Prevent', 'Refuse', 'Taunt', 'Thank', 'Warn',
        ],
        'speaker_list' : ['friends person5', 'Cheyenne', 'Tate', 'Joey', 'Chandler', 'Dina', 'Myrtle', 'Sheldon', 'Garrett', 'Amy', 'Rajesh', 'Justine', 'friends person4',
            'Leonard', 'friends person2', 'Janet', 'Jerry', 'Glenn', 'big bang person1', 'Penny', 'superstore person1', 'superstore person2', 'superstore person4',
            'Sandra', 'friends person1', 'Carol', 'Monica', 'Phoebe', 'Jeff', 'big bang person3', 'Emily', 'Gleen', 'Rachel', 'Howard', 'Adam', 'Ross', 'big bang person2',
            'superstore person3', 'Nico', 'Kelly', 'Bo', 'big bang person4', 'Jonah', 'Bernadette', 'Cody', 'Marcus', 'friends person3'
        ],
        'max_seq_lengths': {
            'text': 50,
            'video': 180,
            'audio': 400,
        },
        'feat_dims': {
            'text': 768,
            'video': 256,
            'audio': 768
        },
    },
    'MELD-DA':{
        'labels': ['a', 'ag', 'ans', 'ap', 'b', 'c', 'dag', 'g', 'o',  'q', 's', 'oth'],
        'max_seq_lengths': {
            'text': 70,
            'video': 250,
            'audio': 520
        },
        'feat_dims': {
            'text': 768,
            'video': 1024,
            'audio': 768
        },
    },
}
