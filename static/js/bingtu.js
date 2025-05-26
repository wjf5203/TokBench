
        var dom = document.getElementById('container');
        console.log(dom);
        var myChart = echarts.init(dom, null, {
        renderer: 'canvas',
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
          }
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
      }
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
      }
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
      }
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
          }
        ]
    },
    {
        name: 'Large\nCharts',
        itemStyle: {
          color: '#039fb8'
        },
        children: [
          {
            name: 'Financial\nCharts',
            value: 1,
            itemStyle: {
              color: '#80a89d'
            }
          },
          {
            name: 'Scientific\nResearch',
            value: 1,
            itemStyle: {
              color: '#7a9bae'
            }
          }
        ]
    }
    ];
    option = {
    series: {
        type: 'sunburst',
        data: data,
        radius: [0, '95%'],
        sort: undefined,
        emphasis: {
        focus: 'ancestor'
        },
        label: {
        textStyle: {
            // color: "#ffffff",
            fontSize: 12,
            // fontWeight: 'bolder',
            fontFamily: 'serif'
        }
        },
        levels: [
        {},
        {
            r0: '16%',
            r: '40%',
            itemStyle: {
            borderRadius: 6,
            borderWidth: 4
            },
            label: {
            rotate: 'tangential'
            }
        },
        {
            r0: '40%',
            r: '75%',
            itemStyle: {
            borderRadius: 6,
            borderWidth: 4
            },
            label: {
            align: 'right'
            }
        },
        {
            r0: '75%',
            r: '77%',
            label: {
            position: 'outside',
            padding: 3,
            silent: false
            },
            itemStyle: {
            // borderWidth: 3
            }
        }
        ]
    }
    };
        if (option && typeof option === 'object') {
        myChart.setOption(option);
        }

window.addEventListener('resize', myChart.resize);