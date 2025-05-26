
    var dom = document.getElementById('container-task');
    var myChart = echarts.init(dom, null, {
      renderer: 'svg',
      useDirtyRect: false
    });
    var app = {};
    
    var option;

    var data = [
  {
    name: 'Remote\nSensing',
    itemStyle: {
      color: '#c94930'
    },
    children: [
      {
        name: 'Vehicle',
        value: 1.1,
        itemStyle: {
          color: '#caa465'
        }
      },
      {
        name: 'Building',
        value: 1.1,
        itemStyle: {
          color: '#dfbd7e'
        }
      },
    ]
  },
  {
    name: 'Video\nMonitoring',
    itemStyle: {
      color: '#ad213e'
    },
    children: [
      {
        name: 'Pedestrian',
        value: 1.1,
        itemStyle: {
          color: '#794752'
        }
      },
      {
        name: 'Vehicle',
        value: 1.1,
        itemStyle: {
          color: '#cc3d41'
        }
      },
    ]
  },
    {
    name: 'General\nScenarios',
    itemStyle: {
      color: '#a87b64'
    },
    children: [
      {
        name: 'Street View',
        value: 1.1,
        itemStyle: {
          color: '#c78869'
        }
      },
      {
        name: 'Flora & Fauna',
        value: 1.1,
        itemStyle: {
          color: '#d4ad12'
        }
      },
    ]
  },
    {
    name: 'Autonomous\nDriving',
    itemStyle: {
      color: '#e65832'
    },
    children: [
      {
        name: 'Traffic Signal',
        value: 1,
        itemStyle: {
          color: '#d45a59'
        }
      },
      {
        name: 'Pedestrian',
        value: 1,
        itemStyle: {
          color: '#da5c1f'
        }
      },
      {
        name: 'OOD Object',
        value: 1,
        itemStyle: {
          color: '#ae341f'
        }
      },
      {
        name: 'Vehicle',
        value: 1,
        itemStyle: {
          color: '#d78823'
        }
      },
    ]
  },
  {
    name: 'Dense\nText',
    itemStyle: {
      color: '#187a2f'
    },
    children: [
      {
        name: 'Books & Posters',
        value: 1,
        itemStyle: {
          color: '#718933'
        }
      },
      {
        name: 'Daily Life Scenes',
        value: 1,
        itemStyle: {
          color: '#a2b029'
        }
      },
    ]
  },
  {
    name: 'Large\nCharts',
    itemStyle: {
      color: '#039fb8'
    },
    children: [
      {
        name: 'Financial Charts',
        value: 1,
        itemStyle: {
          color: '#80a89d'
        }
      },
      {
        name: 'Scientific Research',
        value: 1,
        itemStyle: {
          color: '#7a9bae'
        }
      },
    ]
  },
];
option = {
  textStyle:{
    fontStyle: 'normal',
    fontFamily: "serif",
    fontSize: 17,
    width: 5,
    overflow: 'breakAll'
  },
  series: {
    type: 'sunburst',
    data: data,
    radius: ['15%', '80%'],
    sort: undefined,
    emphasis: {
      focus: 'ancestor'
    },
    levels: [
      {
      },
      {
        r0: '15%',
        r: '35%',
        itemStyle: {
          borderWidth: 2,
           borderRadius: 15,
           opacity: 0.75,
           shadowColor: 'grey',
          // shadowBlur: 20
        },
        label: {
          rotate: 'tangential',
          align: 'center',
          fontSize: 13,
          fontWeight: "bold"
        }
      },
      {
        r0: '35%',
        r: '70%',
        label: {
          align: 'center',
          fontSize: 10,
          fontWeight: "bold"
        },
        itemStyle: {
          borderWidth: 4,
           borderRadius: 21,
           shadowColor: 'grey',
          // shadowBlur: 40
        },
      },
      {
        r0: '70%',
        r: '72%',
        label: {
          position: 'outside',
          padding: 3,
          silent: true
        },
        itemStyle: {
          borderWidth: 5
        }
      }
    ]
  }
};

    if (option && typeof option === 'object') {
      myChart.setOption(option);
    }

    window.addEventListener('resize', myChart.resize);
